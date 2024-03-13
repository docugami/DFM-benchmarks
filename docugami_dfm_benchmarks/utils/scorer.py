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
            avg_f1 = np.mean(scores[metric]) * 100
        else:
            scores[metric] /= total_rows

    scores["avg_f1"] = avg_f1


def _compute_scores_for_column(
    gt_annotations: list[str], model_outputs: list[str]
) -> dict[str, Any]:
    """
    Computes the scores for a single column given lists of ground truth annotations and model outputs.
    """
    scores = {
        f"{SIM_TITLE}0.8": 0,
        f"{SIM_TITLE}0.6": 0,
        "exact_match": 0,
        "no_output": 0,
        "f1_per_row": [],
    }

    for gt_annotation, model_output in zip(gt_annotations, model_outputs):
        gt_annotation = normalize(gt_annotation)
        model_output = normalize(model_output)

        scores["f1_per_row"].append(compute_f1(gt_annotation, model_output))  # type: ignore

        if gt_annotation == model_output:
            scores["exact_match"] += 1  # type: ignore
        elif not model_output and gt_annotation:
            scores["no_output"] += 1  # type: ignore

        if gt_annotation and model_output:
            similarity = semantic_similarity(gt_annotation, model_output)
            if similarity >= 0.8:
                scores[f"{SIM_TITLE}0.8"] += 1  # type: ignore
            if similarity >= 0.6:
                scores[f"{SIM_TITLE}0.6"] += 1  # type: ignore

    return scores


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
        column_scores = _compute_scores_for_column(gt_annotations, model_outputs)
        _finalize_scores(column_scores, len(data))
        scores[column] = column_scores

    return scores


def score_by_separate_csvs(
    ground_truth_data: list[dict[str, Any]],
    model_output_data: list[dict[str, Any]],
    key_column: Optional[str] = None,
) -> tuple[dict, set, set]:
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
    # Create mappings for ground truth and model output columns from normalized to original names
    gt_columns_normalized = {normalize(key): key for key in ground_truth_data[0].keys()}
    model_columns_normalized = {
        normalize(key): key for key in model_output_data[0].keys()
    }

    # Identify common columns based on normalized names and keep track of the original names for later use
    common_columns_normalized = set(gt_columns_normalized.keys()).intersection(
        model_columns_normalized.keys()
    )

    # Initialize scores dictionary
    scores = {}

    # Prepare sets to track ignored columns based on their original names
    ignored_columns_gt = set(ground_truth_data[0].keys()) - set(
        gt_columns_normalized[norm] for norm in common_columns_normalized
    )
    ignored_columns_model = set(model_output_data[0].keys()) - set(
        model_columns_normalized[norm] for norm in common_columns_normalized
    )

    # Iterate over common columns using the normalized names to facilitate comparison
    for norm_col in common_columns_normalized:
        original_gt_col = gt_columns_normalized[norm_col]
        original_model_col = model_columns_normalized[norm_col]

        gt_annotations = [row[original_gt_col] for row in ground_truth_data]
        model_outputs = [row[original_model_col] for row in model_output_data]

        column_scores = _compute_scores_for_column(gt_annotations, model_outputs)
        _finalize_scores(column_scores, len(ground_truth_data))
        scores[original_gt_col] = (
            column_scores  # Use the ground truth's original column name
        )

    return scores, ignored_columns_gt, ignored_columns_model
