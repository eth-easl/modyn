from pathlib import Path
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs
from modyn.supervisor.internal.pipeline_executor.evaluation_executor import rerun_evaluations_from_path
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

LOG_PATH = Path("/Users/mboether/phd/dynamic-data/dynamic_datasets_dsl/benchmark/sigmod")

def rerun_evaluations_for_device(device, pipeline_logs):
    for pipeline_log in pipeline_logs:
        pipeline_logdir = pipeline_log.parent
        print(f"Running evaluation for {device} on {pipeline_logdir}")
        #rerun_evaluations_from_path(pipeline_logdir)
        print(f"Finished evaluation for {device} on {pipeline_logdir}")

    print(f"Finished all evaluations on {device}")

def rerun_evals():
    pipeline_logs = list(LOG_PATH.glob("**/pipeline.log"))
    relevant_logs = defaultdict(list)
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

        relevant_logs[parsed_log.config.pipeline.evaluation.device].append(pipeline_log)

    print(f"--- Relevant logs: {relevant_logs}")
    input("Press enter to confirm and start evaluations") or None
    print("Starting evaluations")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(relevant_logs.keys()) + 1) as executor:
        futures = {executor.submit(rerun_evaluations_for_device, device, logs): device for device, logs in relevant_logs.items()}
        for future in concurrent.futures.as_completed(futures):
            device = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred while running evaluations for {device}: {e}")


if __name__ == "__main__":
    rerun_evals()
