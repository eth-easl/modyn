import os
from pathlib import Path

from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs

EVAL_DIR_ENV = os.environ.get("MODYN_EVAL_DIR")
if EVAL_DIR_ENV:
    EVAL_DIR = Path(EVAL_DIR_ENV)
else:
    EVAL_DIR = Path(input("Please enter the path to the evaluation directory: "))

assert EVAL_DIR.exists(), f"Evaluation directory does not exist: {EVAL_DIR}"


def list_pipelines() -> dict[int, tuple[str, Path]]:
    """Returns a list of tuples with pipeline_id, optional pipeline_name, and pipeline_dir"""
    pipeline_dirs = [x.name for x in EVAL_DIR.glob("pipeline_*") if x.is_dir()]

    pipelines = {}
    for pipeline in pipeline_dirs:
        pipeline_id = int(pipeline.split("_")[-1])
        pipeline_namefile = EVAL_DIR / pipeline / ".name"
        if pipeline_namefile.exists():
            pipeline_name = pipeline_namefile.read_text()
        else:
            pipeline_name = pipeline

        pipelines[pipeline_id] = (pipeline_name, Path(pipeline))

    return pipelines


def load_pipeline_logs(pipeline_id: int) -> PipelineLogs:
    pipeline_logfile = EVAL_DIR / f"pipeline_{pipeline_id}" / "pipeline.log"
    if not pipeline_logfile.exists():
        raise FileNotFoundError(f"Pipeline log file not found: {pipeline_logfile}")

    return PipelineLogs.model_validate_json(pipeline_logfile.read_text())
