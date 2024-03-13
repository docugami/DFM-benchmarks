from enum import Enum
from typing import Any

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from docugami_dfm_benchmarks.utils.similarity import compute_f1, semantic_similarity
from docugami_dfm_benchmarks.utils.text import normalize

KEY_GT = "Ground Truth"
sim_title = "Similarity@>="


class OutputFormat(str, Enum):
    TSV = "tsv"
    GITHUB_MARKDOWN = "github"


def score_data(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Scores the data in the given input. Assumes data is in the following format:

    data_col_1 | data_col_2 | ... | data_col_n | Ground Truth   | model_col_1 | ... | model_col_n
    -----------|------------|-----|------------|----------------|-------------|-----|------------
    data_x     |  data_y    | ... |  data_z    | label_x        | label_y     | ... | label_z
    ...

    Ignores the data_col_* values, and looks at the columns to the right of Ground Truth.

    Scores all the model_col_* values to the right of the Ground Truth column against the
    Ground Truth column using a few different metrics.
    """
    column_headers = list(data[0].keys())

    try:
        gt_col_index = column_headers.index(KEY_GT)
    except ValueError:
        raise Exception(
            f"Ground truth annotation column not found, expected {KEY_GT} in list {column_headers}"
        )

    # all columns to the right of the GT column are models
    ai_model_headers = column_headers[gt_col_index + 1 :]
    scores: dict[str, dict[str, Any]] = {
        model: {
            f"{sim_title}0.8": 0,
            f"{sim_title}0.6": 0,
            "exact_match": 0,
            "no_output": 0,
            "f1_per_row": [],
        }
        for model in ai_model_headers
    }

    for row in tqdm(data):
        gt_annotation = normalize(row[KEY_GT])
        for model in ai_model_headers:
            model_output = normalize(row[model])

            # Token F1 for this row
            scores[model]["f1_per_row"].append(compute_f1(gt_annotation, model_output))

            if gt_annotation == model_output:
                # Exact match
                scores[model]["exact_match"] += 1
            elif not model_output and gt_annotation:
                # Model output is empty, but ground truth annotation is not
                scores[model]["no_output"] += 1

            if gt_annotation and model_output:
                # Semantic similarity at different thresholds
                similarity = semantic_similarity(gt_annotation, model_output)
                if similarity >= 0.8:
                    scores[model][f"{sim_title}0.8"] += 1
                if similarity >= 0.6:
                    scores[model][f"{sim_title}0.6"] += 1

    total_rows = len(data)

    for model in ai_model_headers:
        scores[model][f"{sim_title}0.8"] = scores[model][f"{sim_title}0.8"] / total_rows
        scores[model][f"{sim_title}0.6"] = scores[model][f"{sim_title}0.6"] / total_rows
        scores[model]["exact_match"] = scores[model]["exact_match"] / total_rows
        scores[model]["no_output"] = scores[model]["no_output"] / total_rows
        scores[model]["avg_f1"] = np.mean(scores[model]["f1_per_row"]) * 100

    return scores


def tabulate_scores(
    scores: dict, output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN
) -> str:
    """Tabulates a set of scores (output of the score() function) into a printable view"""
    headers = [
        "Model",
        "Exact Match",
        f"{sim_title} 0.8",
        f"{sim_title} 0.6",
        "Average F1",
        "No Output",
    ]
    table = []

    for model, metrics in scores.items():
        table.append(
            [
                model,
                metrics["exact_match"],
                metrics[f"{sim_title}0.8"],
                metrics[f"{sim_title}0.6"],
                metrics["avg_f1"],
                metrics["no_output"],
            ]
        )

    return tabulate(
        table, headers=headers, floatfmt=".2f", tablefmt=output_format.value
    )
