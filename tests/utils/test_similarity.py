from docugami_dfm_benchmarks.utils.similarity import compute_f1


def test_compute_f1_exact_match() -> None:
    """
    Test compute_f1 with texts that are exactly the same.
    Expected F1 score should be 1.0, indicating a perfect match.
    """
    text1 = "This is a test"
    text2 = "This is a test"
    expected_f1 = 1.0
    assert compute_f1(text1, text2) == expected_f1


def test_compute_f1_normalized_match() -> None:
    """
    Test compute_f1 with texts that match exactly when normalized.
    This includes removal of articles, ignoring whitespace differences, and case insensitivity.
    Expected F1 score should be 1.0, indicating a perfect match after normalization.
    """
    text1 = "One two a three"
    text2 = "one two   three"
    expected_f1 = 1.0
    assert compute_f1(text1, text2) == expected_f1


def test_compute_f1_no_match() -> None:
    """
    Test compute_f1 with texts that have no matching tokens.
    Expected F1 score should be 0.0, indicating no similarity between the texts.
    """
    text1 = "One two a three"
    text2 = "   four five a six"
    expected_f1 = 0.0
    assert compute_f1(text1, text2) == expected_f1


def test_compute_f1_partial_match() -> None:
    """
    Test compute_f1 with partially overlapping tokens to ensure correct partial matching.
    """
    text1 = "quick brown fox"
    text2 = "lazy brown dog"
    # Expected F1 considering overlap is "brown", with precision = recall = F1 = 1/3
    expected_f1 = 2 * (1 / 3 * 1 / 3) / (1 / 3 + 1 / 3)
    assert compute_f1(text1, text2) == expected_f1


def test_compute_f1_with_empty_strings() -> None:
    """
    Test compute_f1 with one or both strings empty to check edge case handling.
    """
    assert compute_f1("", "") == 1.0  # Both empty, perfect match
    assert compute_f1("quick brown fox", "") == 0.0  # One empty, no match
