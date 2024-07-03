import os
from pathlib import Path

from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs

def process_name(name: str) -> str:
    if name[-4] == 'r':
        return name
    
    return f"{name}_r500"

def group_pipelines_by_name(pipeline_logs_directory: Path) -> dict[str, list[Path]]:
    # find the groups of equivalent pipelines via the .name file

    pipeline_directories = [
        pipeline_logs_directory / d for d in os.listdir(pipeline_logs_directory) if str(d).startswith("pipeline_")
    ]

    pipeline_names: list[tuple[Path, str]] = [
        (d, process_name((d / ".name").read_text())) for d in pipeline_directories if (d / "pipeline.log").exists()
    ]

    pipeline_groups = {name: [d for d, n in pipeline_names if n == name] for name in set(n for _, n in pipeline_names)}
    return pipeline_groups


def load_multiple_logfiles(pipeline_files: list[Path]) -> list[PipelineLogs]:
    """
    Args:
        pipeline_files: list of paths to pipeline log directories (not files!)
    Returns:
        list of PipelineLogs
    """
    logs = [
        PipelineLogs.model_validate_json((pipeline_logfile / "pipeline.log").read_text())
        for pipeline_logfile in pipeline_files
    ]
    return logs
