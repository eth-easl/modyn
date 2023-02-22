import argparse
import logging
import os
import pathlib
import random
import shutil
import time
import json

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

PIPELINES = {"finetune_noreset": "Finetune",
            } # TODO(MaxiBoether): Add our pipelines after defining them
RESULTS = {}

def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Criteo Model Evaluation Script")
    parser_.add_argument(
        "dir", type=pathlib.Path, action="store", help="Path to trained models directory"
    )
    parser_.add_argument(
        "output", type=pathlib.Path, action="store", help="Path to output json file"
    )

    return parser_

def validate_output_file_does_not_exist(output: pathlib.Path):
    if output.exists():
        raise ValueError(f"Output file {output} already exists.")

def validate_model_directory(dir: pathlib.Path):
    for pipeline in PIPELINES:
        pipeline_dir = dir / pipeline.replace(' ', '_')
        models = list(filter(os.path.isdir, os.listdir(pipeline_dir)))
        if len(models) == 0:
            logger.warning(f"Pipeline {pipeline}: No trained model found, ignoring.")
            continue

        if any((not name.isdigit() for name in models)):
            raise ValueError(f"Found invalid (non-numeric) model for pipeline {pipeline}, don't know how to proceed")

        models.sort(key = int)
        latest_model_path


def evaluate(dir: pathlib.Path):
    pass

def write_results_to_file(output: pathlib.Path):
    pass



def main():
    parser = setup_argparser()
    args = parser.parse_args()
    validate_output_file_does_not_exist(args.output)
    validate_model_directory(args.dir)
    evaluate(args.dir)

if __name__ == "__main__":
    main()
