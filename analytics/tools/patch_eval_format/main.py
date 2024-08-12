"""# Motivation

Patches old logfiles to the new evaluation logfile format with batched
evaluations.
"""

from pathlib import Path
from typing import Annotated

import typer

from analytics.app.data.load import list_pipelines
from analytics.tools.patch_eval_format.patch_eval import patch_logfile


def main(
    logs_directory: Annotated[Path, typer.Argument(help="Path to read the pipelines in from")],
    patched_log_dir: Annotated[Path, typer.Argument(help="Path to output the patched pipelines to")],
) -> None:
    # find the groups of equivalent pipelines via the .name file

    pipelines = list_pipelines(logs_directory)

    for pipeline_id, (pipe_name, pipeline_path) in pipelines.items():
        logfile = logs_directory / pipeline_path / "pipeline.log"
        if "rho_loss" in pipe_name:
            print(f"Skipping {logfile}")
            continue
        patched = patch_logfile(logfile)
        (patched_log_dir / pipeline_path).mkdir(parents=True, exist_ok=True)
        (patched_log_dir / pipeline_path / (logfile.stem + ".log")).write_text(patched.model_dump_json(by_alias=True))
        (patched_log_dir / pipeline_path / ".name").write_text((logs_directory / pipeline_path / ".name").read_text())


if __name__ == "__main__":
    typer.run(main)
