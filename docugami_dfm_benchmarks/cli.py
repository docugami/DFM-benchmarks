import csv
import sys
from pathlib import Path
from typing import Optional

import typer

from docugami_dfm_benchmarks.utils.scorer import score_by_column, score_by_separate_csvs
from docugami_dfm_benchmarks.utils.tabulation import OutputFormat, tabulate_scores

app = typer.Typer(
    help="Docugami Foundation Model (DFM) Benchmark evaluation scripts",
    no_args_is_help=True,
)


@app.command()
def eval_by_column(
    csv_file: Path,
    output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN,
) -> None:
    """
    Scores the data in the given input CSV file. Assumes data is in the following format:

    data_col_1 | data_col_2 | ... | data_col_n | Ground Truth   | model_col_1 | ... | model_col_n
    -----------|------------|-----|------------|----------------|-------------|-----|------------
    data_x     |  data_y    | ... |  data_z    | label_x        | label_y     | ... | label_z
    ...

    Ignores the data_col_* values, and looks at the columns to the right of Ground Truth.

    Scores all the model_col_* values to the right of the Ground Truth column against the
    Ground Truth column using a few different metrics.
    """
    with open(csv_file) as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        scores = score_by_column(data)
        table = tabulate_scores(scores, output_format)
        typer.echo(table)


@app.command()
def eval_by_csv(
    ground_truth_csv: Path,
    model_output_csv: Path,
    key_column: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN,
) -> None:

    with open(ground_truth_csv, encoding="utf-8-sig") as gt_file:
        gt_reader = csv.DictReader(gt_file)
        gt_data = [row for row in gt_reader]
        with open(model_output_csv, encoding="utf-8-sig") as model_output_file:
            model_output_reader = csv.DictReader(model_output_file)
            model_output_data = [row for row in model_output_reader]

            (
                scores,
                ignored_columns_gt,
                ignored_columns_model,
                unmatched_gt,
                unmatched_mo,
            ) = score_by_separate_csvs(gt_data, model_output_data, key_column)
            table = tabulate_scores(scores, output_format)
            typer.echo(table)

            typer.echo(
                f"Ignored columns in ground truth CSV (no match in model output): {ignored_columns_gt}"
            )
            typer.echo(
                f"Ignored columns in model output CSV (no match in ground truth): {ignored_columns_model}"
            )

            if key_column:
                typer.echo(
                    f"{len(unmatched_gt)} rows in ground truth did not have matching rows in model output (based on key column {key_column})"
                )
                typer.echo(
                    f"{len(unmatched_mo)} rows in model output did not have matching rows in ground truth (based on key column {key_column})"
                )


def _version_callback(value: bool) -> None:
    """
    Gets the current version number from the Poetry package.
    See: https://typer.tiangolo.com/tutorial/options/version/#fix-with-is_eager
    """
    if value:
        try:
            from importlib_metadata import version
        except ModuleNotFoundError:
            from importlib.metadata import version

        typer.echo(version("docugami_dfm_benchmarks"))
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Prints the version number.",
    )
) -> None:
    pass


if __name__ == "__main__":
    if sys.gettrace() is not None:
        # debugger attached, modify call below and attach
        # eval_by_column(Path("./temp/CSL-Small.csv"))  # nosec
        eval_by_csv(
            Path("./temp/tangible_ground_truth.csv"),
            Path("./temp/tangible_model_output.csv"),
            key_column="COMPLAINT PDF FILE NAME",
        )
    else:
        # proceed as normal
        app()
