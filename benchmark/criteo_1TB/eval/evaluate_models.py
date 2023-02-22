import argparse
import logging
import os
import pathlib
import sys
import json

from evaluation_support import evaluate_model

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

PIPELINES = {"finetune_noreset": "Finetune",
            } # TODO(MaxiBoether): Add our pipelines after defining them
NUM_DAYS = 22

def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Criteo Model Evaluation Script")
    parser_.add_argument(
        "dir", type=pathlib.Path, action="store", help="Path to trained models directory"
    )
    parser_.add_argument(
        "output", type=pathlib.Path, action="store", help="Path to output json file"
    )
    parser_.add_argument(
        "evaluation_data", type=pathlib.Path, action="store", help="Path to evaluation data (day 23)"
    )

    return parser_

def validate_output_file_does_not_exist(output: pathlib.Path):
    if output.exists():
        raise ValueError(f"Output file {output} already exists.")

def validate_model_directory(dir: pathlib.Path) -> dict:
    pipelines_to_evaluate = {}

    for pipeline in PIPELINES:
        pipeline_dir = dir / pipeline.replace(' ', '_')

        if not pipeline_dir.exists():
            logger.warning(f"Did not find pipeline {pipeline}, ignoring.")
            continue

        runs = list(filter(os.path.isdir, os.listdir(pipeline_dir)))
        if len(runs) == 0:
            logger.warning(f"Pipeline {pipeline}: No runs found, ignoring.")
            continue

        if any((not name.isdigit() for name in runs)):
            raise ValueError(f"Found invalid (non-numeric) run for pipeline {pipeline}, don't know how to proceed")

        runs.sort(key = int)
        latest_run_path = pipeline_dir / runs[-1]
        
        for i in range(0, NUM_DAYS):
            model_path = latest_run_path / f"{i}.modyn"
            if not model_path.exists():
                raise ValueError(f"Pipeline {pipeline}: Run {runs[-1]} is invalid, could not find model for day {i}.")

        pipelines_to_evaluate[pipeline] = latest_run_path

    return pipelines_to_evaluate

def evaluate_pipeline(dir: pathlib.Path, evaluation_data: pathlib.Path) -> dict:
    pipeline_data = {}
    for i in range(0, NUM_DAYS):
        model_path = dir / f"{i}.modyn"
        pipeline_data[i] = evaluate_model(model_path, evaluation_data)

    return pipeline_data

def evaluate(pipelines_to_evaluate: dict, evaluation_data: pathlib.Path) -> dict:
    results = {}
    for pipeline, path in pipelines_to_evaluate:
        pipeline_data = evaluate_pipeline(path, evaluation_data)
        results[pipeline] = pipeline_data

    return results

def write_results_to_file(results: dict, output: pathlib.Path):
    with open(output, "w") as output_file:
        json.dump(results, output_file)

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    validate_output_file_does_not_exist(args.output)
    pipelines_to_evaluate = validate_model_directory(args.dir)

    if len(pipelines_to_evaluate) == 0:
        logger.error("Found no pipeline data to evaluate, exiting.")
        sys.exit(-1)

    results = evaluate(pipelines_to_evaluate, args.evaluation_data)

    write_results_to_file(results, args.output)

if __name__ == "__main__":
    main()
