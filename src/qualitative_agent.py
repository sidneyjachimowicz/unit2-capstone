"""
qualitative_agent.py

Semantic search over the company's policy documents using a local Chroma
vector store. Documents are split into sections (by markdown ## headers)
for more precise retrieval and citation than whole-document chunks.
Embeddings are generated locally via Sentence Transformers (no API key
needed). Gemini is used only for the final answer generation step, once
relevant sections have been retrieved.
"""

import os
import re
import glob
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

from src.tokenomics import log_usage

load_dotenv()

DOCS_DIR = os.path.join("data", "docs")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "policy_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
AGENT_NAME = "QualitativeAgent"
TOP_K = 3  # how many chunks to retrieve per query

_embedding_model = None
_chroma_client = None
_gemini_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_chroma_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _chunk_markdown_by_section(text: str, doc_name: str) -> list[dict]:
    """
    Split a markdown document into chunks by ## headers.
    Returns a list of {"text": ..., "doc_name": ..., "section": ...}.
    The top-level # title and any text before the first ## is kept
    as its own "Overview" chunk.
    """
    # Split on lines starting with "## " (level-2 headers)
    parts = re.split(r"(?m)^##\s+", text)

    chunks = []
    # Text before the first "## " is just the "# Title" line — too short and
    # generic to be a useful standalone retrieval unit (short chunks tend to
    # embed near a generic centroid and pollute retrieval for unrelated
    # queries). Fold it into the first real section instead of indexing it
    # on its own.
    title_text = parts[0].strip()

    for idx, part in enumerate(parts[1:]):
        lines = part.strip().split("\n", 1)
        section_title = lines[0].strip()
        section_body = lines[1].strip() if len(lines) > 1 else ""
        full_text = f"{section_title}\n{section_body}".strip()

        if idx == 0 and title_text:
            full_text = f"{title_text}\n\n{full_text}".strip()

        if full_text:
            chunks.append({"text": full_text, "doc_name": doc_name, "section": section_title})

    return chunks


def build_index(docs_dir: str = DOCS_DIR, rebuild: bool = True):
    """
    Read all .md files in docs_dir, chunk them by section, embed each
    chunk locally, and store them in the Chroma collection. Call this
    once before querying, or whenever the docs change.
    """
    model = _get_embedding_model()

    global _chroma_client
    if rebuild:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet, that's fine

    collection = _get_chroma_collection()

    doc_paths = glob.glob(os.path.join(docs_dir, "*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    all_ids, all_texts, all_metadatas = [], [], []
    chunk_counter = 0

    for path in doc_paths:
        doc_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = _chunk_markdown_by_section(text, doc_name)
        for chunk in chunks:
            all_ids.append(f"chunk_{chunk_counter}")
            all_texts.append(chunk["text"])
            all_metadatas.append({"doc_name": chunk["doc_name"], "section": chunk["section"]})
            chunk_counter += 1

    embeddings = model.encode(all_texts).tolist()

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )

    print(f"Indexed {chunk_counter} chunks from {len(doc_paths)} documents.")
    return chunk_counter


def _retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    model = _get_embedding_model()
    collection = _get_chroma_collection()

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "doc_name": results["metadatas"][0][i]["doc_name"],
            "section": results["metadatas"][0][i]["section"],
            "distance": results["distances"][0][i],
        })
    return chunks


def answer_qualitative_question(question: str, top_k: int = TOP_K) -> dict:
    """
    Full pipeline: embed question -> retrieve relevant chunks from Chroma
    -> ask Gemini to answer using only those chunks -> return answer with
    source attribution.
    """
    chunks = _retrieve_relevant_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "success": False,
            "question": question,
            "answer": None,
            "sources": [],
            "error": "No relevant documents found. Has build_index() been run?",
        }

    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['doc_name']} — {c['section']}]\n{c['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about company policy
using ONLY the provided document excerpts below. If the excerpts don't contain
enough information to answer, say so clearly rather than guessing.

Document excerpts:
{context}

Question: {question}

Answer clearly and cite which document/section you drew from.
"""

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )

    usage = response.usage_metadata
    log_usage(
        AGENT_NAME,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )

    sources = [{"doc_name": c["doc_name"], "section": c["section"]} for c in chunks]

    return {
        "success": True,
        "question": question,
        "answer": response.text.strip(),
        "sources": sources,
    }


if __name__ == "__main__":
    # Build the index once, then run a quick manual smoke test.
    build_index()
    result = answer_qualitative_question("What is our company's security policy on passwords?")
    print("\n--- RESULT ---")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])