import numpy as np

from docugami_dfm_benchmarks.utils.scorer import _finalize_scores, score_by_column
from docugami_dfm_benchmarks.utils.similarity import SIM_TITLE


def test_finalize_scores() -> None:
    scores = {"exact_match": 2, "no_output": 1, "f1_per_row": np.array([1, 0.5, 0.75])}
    total_rows = 3
    _finalize_scores(scores, total_rows)
    assert scores["exact_match"] == 2 / 3
    assert scores["no_output"] == 1 / 3
    assert scores["avg_f1"] == np.mean([100, 50, 75])


def test_score_by_column() -> None:
    data = [
        {
            "Ground Truth": "Test sentence.",
            "Model A": "Test sentence.",
            "Model B": "test sentence",
        },
        {
            "Ground Truth": "Another test.",
            "Model A": "A different sentence.",
            "Model B": "",
        },
    ]
    expected_scores = {
        "Model A": {
            "avg_f1": 50.0,
            "exact_match": 0.5,
            "no_output": 0,
            f"{SIM_TITLE}0.8": 0.5,
            f"{SIM_TITLE}0.6": 0.5,
        },
        "Model B": {
            "avg_f1": 50.0,
            "exact_match": 0.5,
            "no_output": 0.5,
            f"{SIM_TITLE}0.8": 0.5,
            f"{SIM_TITLE}0.6": 0.5,
        },
    }
    scores = score_by_column(data)
    for column in expected_scores:
        for metric in expected_scores[column]:
            assert np.isclose(
                scores[column][metric], expected_scores[column][metric], atol=0.01
            )
