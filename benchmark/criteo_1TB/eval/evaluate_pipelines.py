import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from evaluation_support import evaluate_model
from modyn.utils.logging import setup_logging

# TODO(MaxiBoether): Add our pipelines after defining them
PIPELINES = {"models_exp0_finetune": "Finetune"}
NUM_DAYS = 3

logger = setup_logging(__name__)


def validate_model_directory(dir: Path) -> dict:
    pipelines_to_evaluate = {}

    for pipeline in PIPELINES:
        pipeline_dir = dir / pipeline.replace(" ", "_")

        if not pipeline_dir.exists():
            logger.warning(f"Did not find pipeline {pipeline}, ignoring.")
            continue

        runs = [run.name for run in pipeline_dir.iterdir() if os.path.isdir(run)]
        if len(runs) == 0:
            logger.warning(f"Pipeline {pipeline}: No runs found, ignoring.")
            continue

        if any((not name.isdigit() for name in runs)):
            raise ValueError(f"Found invalid (non-numeric) run for pipeline {pipeline}, don't know how to proceed")

        runs.sort(key=int)
        latest_run_path = pipeline_dir / runs[-1]

        logger.info(f"chose {latest_run_path}")

        for i in range(0, NUM_DAYS):
            model_path = latest_run_path / f"{i + 1}.modyn"  # triggers are 1-indexed, hence + 1
            if not model_path.exists():
                raise ValueError(f"Pipeline {pipeline}: Run {runs[-1]} is invalid, could not find model for day {i}.")

        pipelines_to_evaluate[pipeline] = latest_run_path

    return pipelines_to_evaluate


def evaluate_pipeline(dir: Path, evaluation_data: Path) -> dict:
    pipeline_data = {}

    pipeline_data[0] = evaluate_model(None, evaluation_data)  # tests randomly initialized model

    for i in range(0, NUM_DAYS):
        model_path = dir / f"{i + 1}.modyn"  # triggers are 1-indexed, hence + 1
        pipeline_data[i + 1] = evaluate_model(model_path, evaluation_data)

    return pipeline_data


def evaluate(pipelines_to_evaluate: dict, evaluation_data: Path) -> dict:
    results = {}
    for pipeline, path in pipelines_to_evaluate.items():
        pipeline_data = evaluate_pipeline(path, evaluation_data)
        results[pipeline] = pipeline_data

    return results


def write_results_to_file(results: dict, output: Path) -> None:
    with open(output, "w") as output_file:
        json.dump(results, output_file)


def main(
    dir: Annotated[Path, typer.Argument(help="Path to trained models directory")],
    output: Annotated[Path, typer.Argument(help="Path to output json file")],
    evaluation_data: Annotated[Path, typer.Argument(help="Path to evaluation data (day 23)")],
) -> None:
    assert not output.exists(), f"Output file {output} already exists."
    pipelines_to_evaluate = validate_model_directory(dir)

    if len(pipelines_to_evaluate) == 0:
        logger.error("Found no pipeline data to evaluate, exiting.")
        sys.exit(-1)

    results = evaluate(pipelines_to_evaluate, evaluation_data)

    write_results_to_file(results, output)


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
