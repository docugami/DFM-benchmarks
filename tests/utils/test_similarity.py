import pytest

from docugami_dfm_benchmarks.utils.similarity import compute_f1


@pytest.mark.parametrize(
    "text1,text2,expected_f1",
    [
        ("This is a test", "This is a test", 1.0),  # Exact match
        ("One two a three", "one two   three", 1.0),  # Exact match modulo article, whitespace and casing
        ("One two a three", "   four five a six", 0.0),  # No match
    ],
)
def test_compute_f1(text1: str, text2: str, expected_f1: float) -> None:
    assert compute_f1(text1, text2) == expected_f1
