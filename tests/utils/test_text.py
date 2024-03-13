import pytest
from docugami_dfm_benchmarks.utils.text import normalize, get_tokens


def test_normalize_basic():
    """Test normalization on a simple string."""
    assert normalize("This is an example.") == "this is example"


def test_normalize_with_punctuation():
    """Test normalization removes punctuation."""
    assert normalize("Hello, world!") == "hello world"


def test_normalize_with_articles():
    """Test normalization removes articles 'a', 'an', 'the'."""
    assert (
        normalize("A quick brown fox jumps over the lazy dog.")
        == "quick brown fox jumps over lazy dog"
    )


def test_normalize_with_extra_whitespace():
    """Test normalization removes extra whitespace."""
    assert normalize("  This    is  a  test.  ") == "this is test"


def test_get_tokens_empty():
    """Test get_tokens returns an empty list for empty input."""
    assert get_tokens("") == []


def test_get_tokens_basic():
    """Test get_tokens on a simple string."""
    assert get_tokens("This is a test.") == ["this", "is", "test"]


def test_get_tokens_complex():
    """Test get_tokens with punctuation and extra whitespace."""
    expected = [
        "this",
        "is",
        "complex",
        "test",
        "with",
        "punctuation",
        "and",
        "extra",
        "whitespace",
    ]
    assert (
        get_tokens(
            "This, is a complex test. With punctuation! And... extra whitespace?"
        )
        == expected
    )
