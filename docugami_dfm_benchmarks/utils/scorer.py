from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from docugami_dfm_benchmarks.utils.similarity import (
    SIM_TITLE,
    compute_f1,
    semantic_similarity,
)
from docugami_dfm_benchmarks.utils.text import normalize

KEY_GT = "Ground Truth"


def _finalize_scores(scores: dict[str, Any], total_rows: int) -> None:
    """
    Normalizes scores by the total number of rows and calculates the average F1 score.

    Parameters:
    - scores: The scores dictionary for a single column.
    - total_rows: The total number of rows over which scores were computed.

    Modifies the scores dictionary in-place to include normalized metrics and the average F1 score.
    """
    avg_f1 = 0
    for metric in list(scores):
        if metric == "f1_per_row":
            avg_f1 = np.mean(scores[metric])
        else:
            scores[metric] /= total_rows

    scores["avg_f1"] = avg_f1


def _initialize_score_structure() -> dict:
    """Initializes the structure for storing scores."""
    return {
        f"{SIM_TITLE}0.8": 0,
        f"{SIM_TITLE}0.6": 0,
        "exact_match": 0,
        "no_output": 0,
        "f1_per_row": [],
    }


def _update_scores(score_struct: dict, gt_annotation: str, model_output: str) -> None:
    """
    Updates the score structure based on a single row's GT and model output, including semantic similarity.
    """
    # Normalize the inputs (Normalization may already be done before this call, depending on the flow)
    gt_annotation = normalize(gt_annotation)
    model_output = normalize(model_output)

    # Compute F1 score and update
    score_struct["f1_per_row"].append(compute_f1(gt_annotation, model_output))

    # Check for exact matches
    if gt_annotation == model_output:
        score_struct["exact_match"] += 1
    elif not model_output and gt_annotation:
        # Consider cases where the model output is empty but there is a GT annotation
        score_struct["no_output"] += 1

    # Calculate semantic similarity if both GT and model outputs are non-empty
    if gt_annotation and model_output:
        similarity = semantic_similarity(gt_annotation, model_output)
        if similarity >= 0.8:
            score_struct[f"{SIM_TITLE}0.8"] += 1
        if similarity >= 0.6:
            score_struct[f"{SIM_TITLE}0.6"] += 1


def _finalize_all_scores(scores: dict, total_matches: int) -> None:
    """Finalizes all score structures within the scores dict."""
    for score_struct in scores.values():
        _finalize_scores(score_struct, total_matches)


def score_by_column(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Scores the data provided in a single CSV, comparing model outputs directly against
    a ground truth column. Assumes a specific CSV format where one column specifies the
    ground truth, and all subsequent columns are model outputs to be scored against this ground truth.

    Parameters:
    - data: List of dictionaries representing rows from the CSV. Each dictionary corresponds to a row,
            with keys as column headers.

    Returns:
    - A dictionary of scores for each model output column, including metrics such as similarity thresholds,
      exact match, no output, and average F1 score.
    """
    data_columns = list(data[0].keys())

    try:
        gt_col_index = data_columns.index(KEY_GT)
    except ValueError:
        raise Exception(
            f"Ground truth annotation column not found, expected {KEY_GT} in list {data_columns}"
        )

    # all columns to the right of the GT column are considered models
    model_columns = data_columns[gt_col_index + 1 :]
    scores = {}

    for column in tqdm(model_columns):
        gt_annotations = [normalize(row[KEY_GT]) for row in data]
        model_outputs = [normalize(row[column]) for row in data]

        # Initialize the score structure for this column
        column_scores = _initialize_score_structure()

        # Update scores for each row
        for gt_annotation, model_output in zip(gt_annotations, model_outputs):
            _update_scores(column_scores, gt_annotation, model_output)

        # Finalize scores by calculating average F1 and normalizing metrics
        _finalize_scores(column_scores, len(data))

        scores[column] = column_scores

    return scores


def score_by_separate_csvs(
    ground_truth_data: list[dict[str, Any]],
    model_output_data: list[dict[str, Any]],
    key_column: Optional[str] = None,
) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """
    Scores model output against ground truth data when provided in separate CSVs.
    Each CSV should have columns with identical names for comparison. This function
    computes scores on a per-column basis for all common columns found in both CSVs.

    Assumes that each row in the ground truth CSV corresponds to the same row in the
    model output CSV. Columns not present in both CSVs are ignored, and a warning
    is logged.

    Parameters:
    - ground_truth_data: List of dictionaries representing rows from the ground truth CSV.
    - model_output_data: List of dictionaries representing rows from the model output CSV.

    Returns:
    - A dictionary of scores for each common column.
    """
    gt_columns_normalized = {normalize(key): key for key in ground_truth_data[0].keys()}
    model_columns_normalized = {
        normalize(key): key for key in model_output_data[0].keys()
    }

    # Identify common columns based on normalized names and keep track of the original names for later use
    common_columns_normalized = set(gt_columns_normalized.keys()).intersection(
        model_columns_normalized.keys()
    )

    scores = {}
    ignored_columns_gt = set(ground_truth_data[0].keys()) - set(
        gt_columns_normalized[norm] for norm in common_columns_normalized
    )
    ignored_columns_model = set(model_output_data[0].keys()) - set(
        model_columns_normalized[norm] for norm in common_columns_normalized
    )

    if key_column:
        gt_keyed_data = {row[key_column]: row for row in ground_truth_data}
        mo_keyed_data = {row[key_column]: row for row in model_output_data}
        matched_rows_set = set(gt_keyed_data.keys()).intersection(mo_keyed_data.keys())
        unmatched_gt = set(gt_keyed_data.keys()) - matched_rows_set
        unmatched_mo = set(mo_keyed_data.keys()) - matched_rows_set
        matched_rows = list(
            matched_rows_set
        )  # Ensure this is always a list for consistency
    else:
        matched_rows = list(range(len(ground_truth_data)))  # type: ignore
        unmatched_gt = unmatched_mo = set()

    for match in matched_rows:
        if key_column:
            gt_row = gt_keyed_data[match]
            mo_row = mo_keyed_data[match]
        else:
            gt_row = ground_truth_data[int(match)]
            mo_row = model_output_data[int(match)]

        for norm_col in common_columns_normalized:
            original_gt_col = gt_columns_normalized[norm_col]
            original_model_col = model_columns_normalized[norm_col]
            if original_gt_col in gt_row and original_model_col in mo_row:
                gt_annotation = gt_row[original_gt_col]
                model_output = mo_row[original_model_col]
                if original_gt_col not in scores:
                    scores[original_gt_col] = _initialize_score_structure()
                _update_scores(scores[original_gt_col], gt_annotation, model_output)

    _finalize_all_scores(scores, len(matched_rows))

    return (
        scores,
        sorted(ignored_columns_gt),
        sorted(ignored_columns_model),
        sorted(unmatched_gt),
        sorted(unmatched_mo),
    )
