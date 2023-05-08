"""
Copyright (c) Docugami Inc.
"""

import csv
from pathlib import Path
import sys

from typing import Optional
import typer

from docugami.dfm_benchmarks.scorer import score_data, tabulate_scores


app = typer.Typer(
    help="Benchmarks for Business Document Foundation Models",
    no_args_is_help=True,
)


@app.command()
def eval(csv_file: Path):
    with open(csv_file) as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        scores = score_data(data)
        table = tabulate_scores(scores)  # type: ignore
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
):
    pass


if __name__ == "__main__":
    if sys.gettrace() is not None:
        # debugger attached, modify call below and attach
        eval("./temp/CSL-Small.csv")
    else:
        # proceed as normal
        app()
