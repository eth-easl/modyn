from pathlib import Path
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs
from tqdm import tqdm
import shutil

from modyn.utils.utils import current_time_millis

LOG_PATH = Path("/Users/mboether/phd/dynamic-data/dynamic_datasets_dsl/benchmark/sigmod")

def merge_logs():
    pipeline_logs = list(LOG_PATH.glob("**/pipeline.log"))
    for pipeline_log in tqdm(pipeline_logs, total=len(pipeline_logs), desc="Parsing pipeline logs"):
        try:
            parsed_log = PipelineLogs.model_validate_json(pipeline_log.read_text())
        except:
            print(f"Skipping file {pipeline_log} due to invalid format")
            continue

        stage_runs = parsed_log.supervisor_logs.stage_runs
        contains_evals = any(stage_run.id == "EVALUATE_SINGLE" for stage_run in  stage_runs)

        if contains_evals:
            print(f"Skipping {pipeline_log} since it contains evaluations already")
            continue
        
        potential_evals = pipeline_log.parent / pipeline_log.parent.name / "pipeline.log"

        if not potential_evals.exists():
            print(f"Did not find results for log file {pipeline_log} at {potential_evals}. Skipping.")
            continue

        try:
            parsed_evals = PipelineLogs.model_validate_json(potential_evals.read_text())
        except:
            print(f"Skipping file {potential_evals} for file {pipeline_log} due to errors while parsing")
            continue

        stage_runs = parsed_evals.supervisor_logs.stage_runs
        contains_evals = any(stage_run.id == "EVALUATE_SINGLE" for stage_run in  stage_runs)

        if not contains_evals:
            print(f"Skipping file {potential_evals} for file {pipeline_log} since it does not contain evaluatoins")
            continue

        # Backup old files
        shutil.copy(pipeline_log, pipeline_log.parent / f"pipeline.log.backup{current_time_millis()}")
        shutil.copy(potential_evals, potential_evals.parent / f"pipeline.log.backup{current_time_millis()}")

        parsed_log.supervisor_logs.stage_runs.extend(parsed_evals.supervisor_logs.stage_runs)
        pipeline_log.write_text(parsed_log.model_dump_json(by_alias=True))

        print(f"Merged {pipeline_log} and {potential_evals}.")

if __name__ == "__main__":
    merge_logs()
