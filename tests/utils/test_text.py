from docugami_dfm_benchmarks.utils.text import get_tokens, normalize


def test_normalize_basic() -> None:
    """Test normalization on a simple string."""
    assert normalize("This is an example.") == "this is example"


def test_normalize_with_punctuation() -> None:
    """Test normalization removes punctuation."""
    assert normalize("Hello, world!") == "hello world"


def test_normalize_with_articles() -> None:
    """Test normalization removes articles 'a', 'an', 'the'."""
    assert (
        normalize("A quick brown fox jumps over the lazy dog.")
        == "quick brown fox jumps over lazy dog"
    )


def test_normalize_with_extra_whitespace() -> None:
    """Test normalization removes extra whitespace."""
    assert normalize("  This    is  a  test.  ") == "this is test"


def test_normalize_with_special() -> None:
    """Test normalization with special chars."""
    assert normalize("AMENDMENT_NUMBER") == "amendment number"


def test_get_tokens_empty() -> None:
    """Test get_tokens returns an empty list for empty input."""
    assert get_tokens("") == []


def test_get_tokens_basic() -> None:
    """Test get_tokens on a simple string."""
    assert get_tokens("This is a test.") == ["this", "is", "test"]


def test_get_tokens_complex() -> None:
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
