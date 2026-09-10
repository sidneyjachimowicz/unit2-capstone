"""
test_gemini_connection.py

Standalone sanity check that your GEMINI_API_KEY is valid and the
google-genai SDK is talking to the API correctly. Run this before
building anything else that depends on Gemini.

Usage:
    python scripts/test_gemini_connection.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found. Check your .env file.")
    exit(1)

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly one word: 'connected'",
)

print("Gemini response:", response.text.strip())
print("\nUsage metadata:", response.usage_metadata)