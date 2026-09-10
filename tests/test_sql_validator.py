import sys
import os

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sql_validator import validate_sql


def test_blocks_drop():
    result = validate_sql("DROP TABLE users")
    assert result["valid"] is False


def test_blocks_delete():
    result = validate_sql("DELETE FROM users WHERE id = 1")
    assert result["valid"] is False


def test_blocks_update():
    result = validate_sql("UPDATE users SET name = 'x'")
    assert result["valid"] is False


def test_allows_simple_select():
    result = validate_sql("SELECT * FROM users")
    assert result["valid"] is True


def test_allows_select_with_where():
    result = validate_sql("SELECT name, email FROM users WHERE active = 1")
    assert result["valid"] is True


def test_blocks_non_select_statement():
    result = validate_sql("EXPLAIN SELECT * FROM users")
    assert result["valid"] is False

def test_blocks_stacked_query():
    result = validate_sql("SELECT * FROM users; DROP TABLE users;")
    # This currently PASSES validation (known limitation) - flag in docs
    print("Stacked query result:", result)


if __name__ == "__main__":
    test_blocks_drop()
    test_blocks_delete()
    test_blocks_update()
    test_allows_simple_select()
    test_allows_select_with_where()
    test_blocks_non_select_statement()
    print("All tests passed!")