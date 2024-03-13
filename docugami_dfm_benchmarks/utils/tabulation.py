from enum import Enum

from tabulate import tabulate

from docugami_dfm_benchmarks.utils.similarity import SIM_TITLE


class OutputFormat(str, Enum):
    TSV = "tsv"
    GITHUB_MARKDOWN = "github"


def tabulate_scores(
    scores: dict, output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN
) -> str:
    """Tabulates a set of scores (output of the score() function) into a printable view"""
    headers = [
        "Column",
        "Exact Match",
        f"{SIM_TITLE} 0.8",
        f"{SIM_TITLE} 0.6",
        "Average F1",
        "No Output",
    ]
    table = []

    for model, metrics in scores.items():
        table.append(
            [
                model,
                metrics["exact_match"],
                metrics[f"{SIM_TITLE}0.8"],
                metrics[f"{SIM_TITLE}0.6"],
                metrics["avg_f1"],
                metrics["no_output"],
            ]
        )

    return tabulate(
        table, headers=headers, floatfmt=".2f", tablefmt=output_format.value
    )
