import csv
import sys
from pathlib import Path
from typing import Optional

import typer

from docugami_dfm_benchmarks.utils.scorer import OutputFormat, score_data, tabulate_scores

app = typer.Typer(
    help="Benchmarks for Business Document Foundation Models",
    no_args_is_help=True,
)


@app.command()
def eval(
    csv_file: Path,
    output_format: OutputFormat = OutputFormat.GITHUB_MARKDOWN,
) -> None:
    with open(csv_file) as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        scores = score_data(data)
        table = tabulate_scores(scores, output_format)
        typer.echo(table)


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
        eval(Path("./temp/CSL-Small.csv"))  # nosec
    else:
        # proceed as normal
        app()
