import numpy as np

from docugami_dfm_benchmarks.utils.scorer import (
    _finalize_scores,
    score_by_column,
    score_by_separate_csvs,
)
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


def test_score_by_separate_csvs_aligned() -> None:
    ground_truth_data = [
        {
            "Column1": "Test sentence.",
            "Column2": "Another test.",
            "Unique GT column": "xyz",
        },
        {
            "Column1": "Second sentence.",
            "Column2": "Yet another test.",
            "Unique GT column": "xyz",
        },
    ]
    model_output_data = [
        {
            "Column1": "Test sentence.",
            "Column2": "",
            "Unique MO column": "abc",
        },
        {
            "Column1": "A different second sentence.",
            "Column2": "Yet another test.",
            "Unique MO column": "abc",
        },
    ]
    expected_scores = {
        "Column1": {
            "avg_f1": 90.0,
            "exact_match": 0.5,
            "no_output": 0,
            f"{SIM_TITLE}0.8": 1.0,
            f"{SIM_TITLE}0.6": 1.0,
        },
        "Column2": {
            "avg_f1": 50.0,  # One exact match, one no_output
            "exact_match": 0.5,
            "no_output": 0.5,
            f"{SIM_TITLE}0.8": 0.5,
            f"{SIM_TITLE}0.6": 0.5,
        },
    }
    scores, ignored_columns_gt, ignored_columns_model, unmatched_gt, unmatched_mo = (
        score_by_separate_csvs(ground_truth_data, model_output_data)
    )

    assert ignored_columns_gt == ["Unique GT column"]
    assert ignored_columns_model == ["Unique MO column"]

    assert not unmatched_gt
    assert not unmatched_mo

    for column in expected_scores:
        for metric in expected_scores[column]:
            assert np.isclose(
                scores[column][metric], expected_scores[column][metric], atol=0.01
            ), f"Failed on {column} {metric}: expected {expected_scores[column][metric]}, got {scores[column][metric]}"


def test_score_by_separate_csvs_with_key_column() -> None:
    # Test data includes a key column named "ID" for matching rows across CSVs
    ground_truth_data = [
        {
            "ID": "1",
            "Column1": "Test sentence.",
            "Column2": "Another test.",
            "Unique GT column": "xyz",
        },
        {
            "ID": "2",
            "Column1": "Second sentence.",
            "Column2": "Yet another test.",
            "Unique GT column": "xyz",
        },
        {
            "ID": "3",
            "Column1": "Third unmatched GT sentence.",
            "Column2": "Unmatched GT test.",
            "Unique GT column": "xyz",
        },
    ]
    model_output_data = [
        {
            "ID": "2",
            "Column1": "A different second sentence.",
            "Column2": "Yet another test.",
            "Unique MO column": "abc",
        },
        {
            "ID": "1",
            "Column1": "Test sentence.",
            "Column2": "",
            "Unique MO column": "abc",
        },
        {
            "ID": "4",
            "Column1": "Fourth unmatched MO sentence.",
            "Column2": "Unmatched MO test.",
            "Unique MO column": "abc",
        },
    ]
    expected_scores = {
        "Column1": {
            "avg_f1": 90.0,  # Considering matched rows only
            "exact_match": 0.5,
            "no_output": 0,
            f"{SIM_TITLE}0.8": 1.0,
            f"{SIM_TITLE}0.6": 1.0,
        },
        "Column2": {
            "avg_f1": 50.0,  # One exact match, one no_output, considering only matched rows
            "exact_match": 0.5,
            "no_output": 0.5,
            f"{SIM_TITLE}0.8": 0.5,
            f"{SIM_TITLE}0.6": 0.5,
        },
    }
    key_column = "ID"
    scores, ignored_columns_gt, ignored_columns_model, unmatched_gt, unmatched_mo = (
        score_by_separate_csvs(
            ground_truth_data, model_output_data, key_column=key_column
        )
    )

    # Test for ignored columns
    assert ignored_columns_gt == ["Unique GT column"]
    assert ignored_columns_model == ["Unique MO column"]

    # Test for mismatched rows
    assert unmatched_gt == ["3"]
    assert unmatched_mo == ["4"]

    # Test scores for matched rows
    for column in expected_scores:
        for metric in expected_scores[column]:
            assert np.isclose(
                scores[column][metric], expected_scores[column][metric], atol=0.01
            ), f"Failed on {column} {metric}: expected {expected_scores[column][metric]}, got {scores[column][metric]}"
