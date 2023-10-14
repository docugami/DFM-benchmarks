"""
Copyright (c) Docugami Inc.
"""

import collections
from enum import Enum
import re
import string
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate
from tqdm import tqdm


KEY_GT = "Ground Truth"
sim_title = "Similarity@>="


class OutputFormat(str, Enum):
    TSV = "tsv"
    GITHUB_MARKDOWN = "github"


embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def _semantic_similarity(text1, text2):
    """Compute semantic similarity (cosine) between embeddings of given texts."""
    embedding_1 = embedding_model.encode(text1, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()  # type: ignore


def _normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _get_tokens(s):
    """Gets normalized tokens from the given string."""
    if not s:
        return []
    return _normalize(s).split()


def _compute_f1(text1, text2) -> float:
    gold_toks = _get_tokens(text1)
    pred_toks = _get_tokens(text2)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def score_data(data: List[Dict]):
    """
    Scores the data in the given input. Assumes data is in the following format:

    data_col_1 | data_col_2 | ... | data_col_n | Ground Truth   | model_col_1 | ... | model_col_n
    -----------|------------|-----|------------|----------------|-------------|-----|------------
    data_x     |  data_y    | ... |  data_z    | label_x        | label_y     | ... | label_z
    ...

    Ignores the data_col_* values, and looks at the columns to the right of Human. Scores
    all the values for models to the right of the Ground Truth column against the ground truth
    using a few different metrics.
    """
    column_headers = list(data[0].keys())

    try:
        gt_col_index = column_headers.index(KEY_GT)
    except ValueError:
        raise Exception(f"Ground truth annotation column not found, expected {KEY_GT} in list {column_headers}")

    ai_model_headers = column_headers[gt_col_index:]  # all columns to the right of the GT column are models
    scores = {
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
        gt_annotation = _normalize(row[KEY_GT])
        for model in ai_model_headers:
            model_output = _normalize(row[model])

            # Token F1 for this row
            scores[model]["f1_per_row"].append(_compute_f1(gt_annotation, model_output))

            if gt_annotation == model_output:
                # Exact match
                scores[model]["exact_match"] += 1
            elif not model_output and gt_annotation:
                # Model output is empty, but human annotation is not
                scores[model]["no_output"] += 1

            if gt_annotation and model_output:
                # Semantic similarity at different thresholds
                similarity = _semantic_similarity(gt_annotation, model_output)
                if similarity >= 0.8:
                    scores[model][f"{sim_title}0.8"] += 1
                if similarity >= 0.6:
                    scores[model][f"{sim_title}0.6"] += 1

    total_rows = len(data)

    for model in ai_model_headers:
        scores[model][f"{sim_title}0.8"] /= total_rows  # type: ignore
        scores[model][f"{sim_title}0.6"] /= total_rows  # type: ignore
        scores[model]["exact_match"] /= total_rows  # type: ignore
        scores[model]["no_output"] /= total_rows  # type: ignore
        scores[model]["avg_f1"] = 100 * np.mean(scores[model]["f1_per_row"])  # type: ignore

    return scores


def tabulate_scores(scores: Dict, output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN):
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

    return tabulate(table, headers=headers, floatfmt=".2f", tablefmt=output_format.value)
