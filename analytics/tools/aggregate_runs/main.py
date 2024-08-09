"""# Motivation

We want to increase the confidence in our pipeline run results by
running the same experiment pipelines with different seeds.

This yields different evaluation metrics. In consequence, we want to
aggregate (e.g. mean, median) the evaluation metrics over runs.
"""

from pathlib import Path
from typing import Annotated

import typer

from analytics.tools.aggregate_runs.core_aggregation import merge_files_for_equivalence_group
from analytics.tools.aggregate_runs.dir_utils import group_pipelines_by_name


def main(
    logs_directory: Annotated[Path, typer.Argument(help="Path to read the pipelines in from")],
    aggregated_log_dir: Annotated[Path, typer.Argument(help="Path to output the aggregated pipelines to")],
    pipeline_name: Annotated[
        str | None,
        typer.Option(
            help=(
                "If not all pipelines should be aggregated, specify the name of the "
                "pipeline to aggregate (as specified in the .name file)"
            )
        ),
    ] = None,
) -> None:
    # find the groups of equivalent pipelines via the .name file

    pipeline_groups = group_pipelines_by_name(logs_directory)

    for group_name, group_pipelines in pipeline_groups.items():
        if pipeline_name is not None and group_name != pipeline_name:
            continue
        merge_files_for_equivalence_group(group_pipelines, output_directory=aggregated_log_dir)


if __name__ == "__main__":
    typer.run(main)
