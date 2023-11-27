import json
import logging
import pathlib
import time
from typing import Optional

import enlighten
from modynclient.client.internal.grpc_handler import GRPCHandler
from modynclient.client.internal.utils import EvaluationStatusTracker, TrainingStatusTracker

POLL_TIMEOUT = 2

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        client_config: dict,
        pipeline_config: dict,
        eval_directory: pathlib.Path,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> None:
        self.client_config = client_config
        self.pipeline_config = pipeline_config
        self.eval_directory = eval_directory
        self.start_replay_at = start_replay_at
        self.stop_replay_at = stop_replay_at
        self.maximum_triggers = maximum_triggers

        self.grpc = GRPCHandler(client_config)
        self.pipeline_id: Optional[int] = None
        self.training_status_tracker: Optional[TrainingStatusTracker] = None

        self.progress_mgr = enlighten.get_manager()
        self.status_bar = self.progress_mgr.status_bar(
            status_format="Modyn{fill}Current Task: {demo}{fill}{elapsed}",
            color="bold_underline_bright_white_on_lightslategray",
            justify=enlighten.Justify.CENTER,
            demo="Initializing",
            autorefresh=True,
            min_delta=0.5,
        )
        self.pbar: Optional[enlighten.Counter] = None
        self.evaluations: dict[int, EvaluationStatusTracker] = {}

    def start_pipeline(self) -> bool:
        logger.info(f"model id: {self.pipeline_config['model']['id']}, maximum_triggers: {self.maximum_triggers}")
        res = self.grpc.start_pipeline(
            self.pipeline_config,
            self.eval_directory,
            self.start_replay_at,
            self.stop_replay_at,
            self.maximum_triggers
        )
        if "exception" in res:
            logger.info(f"Pipeline <{res['pipeline_id']}> failed with error {res['exception']}.")
            return False
        else:
            self.pipeline_id = res['pipeline_id']
            logger.info(f"Pipeline <{self.pipeline_id}> started.")
            return True

    def _monitor_pipeline_progress(self, msg: dict) -> None:
        if msg["log"]:
            logger.info(msg)

        msg_type = msg['msg_type']

        demo = f"Pipeline <{self.pipeline_id}> {msg['stage']}"

        if msg_type == "general":
            pass
        elif msg_type == "id":
            submsg = msg['id_msg']
            demo += f" ({submsg['id_type']} = {submsg['id']})"
        elif msg_type == "dataset":
            demo += f" (dataset = {msg['dataset_msg']['id']})"
        elif msg_type == "counter":
            submsg = msg['counter_msg']
            if submsg["action"] == "create":
                self.pbar = self.progress_mgr.counter(
                    total=submsg["create_params"]["new_data_len"],
                    desc=f"[Pipeline {self.pipeline_id}] Processing New Samples",
                    unit="samples"
                )
            elif submsg["action"] == "update":
                assert self.pbar is not None
                self.pbar.update(submsg["update_params"]["batch_size"])
            elif submsg["action"] == "close":
                assert self.pbar is not None
                self.pbar.clear(flush=True)
                self.pbar.close(clear=True)
        else:
            logger.error(f"unknown msg_type {msg_type} for running pipeline")

        self.status_bar.update(demo=demo)

    def _monitor_training_progress(self, msg: dict) -> None:
        id = msg["id"]

        if msg["action"] == "create_tracker":
            params = msg["training_create_tracker_params"]
            self.training_status_tracker = TrainingStatusTracker(
                self.progress_mgr, id, params["total_samples"],  params["status_bar_scale"]
            )
        elif msg["action"] == "progress_counter":
            params = msg["training_progress_counter_params"]
            self.training_status_tracker.progress_counter(
                params["samples_seen"], params["downsampling_samples_seen"], params["is_training"]
            )
        elif msg["action"] == "close_counter":
            self.training_status_tracker.close_counter()
            

    def _monitor_evaluation_progress(self, msg: dict) -> None:
        id = msg["id"]
        if msg["action"] == "create_tracker":
            params = msg["eval_create_tracker_params"]
            self.evaluations[id] = EvaluationStatusTracker(
                params["dataset_id"], params["dataset_size"]
            )
        elif msg["action"] == "create_counter":
            params = msg["eval_create_counter_params"]
            self.evaluations[id].create_counter(
                self.progress_mgr, params["training_id"], id
            )
        elif msg["action"] == "progress_counter":
            params = msg["eval_progress_counter_params"]
            self.evaluations[id].progress_counter(params["total_samples_seen"])
        elif msg["action"] == "end_counter":
            params = msg["eval_end_counter_params"]
            self.evaluations[id].end_counter(params["error"])
            if params["error"]:
                logger.info(f"Evaluation {id} failed with error: {params['exception_msg']}")


    def poll_pipeline_status(self) -> None:
        res = self.grpc.get_pipeline_status(self.pipeline_id)

        while res["status"] == "running":
            # print(json.dumps(res, sort_keys=True, indent=2))
            if "pipeline_stage" in res:
                self._monitor_pipeline_progress(res["pipeline_stage"])

            if "training_status" in res:
                self._monitor_training_progress(res["training_status"])
            
            if "eval_status" in res:
                self._monitor_evaluation_progress(res["eval_status"])

            time.sleep(POLL_TIMEOUT)

            res = self.grpc.get_pipeline_status(self.pipeline_id)

        if res["status"] == "exit":
            if res["pipeline_stage"]["exit_msg"]["exitcode"] == 0:
                logger.info(f"Pipeline <{self.pipeline_id}> finished successfully.")
            else:
                logger.info(
                    f"Pipeline <{self.pipeline_id}> exited with error {res['pipeline_stage']['exit_msg']['exception']}"
                )
        elif res["status"] == "not found":
            logger.info(f"Pipeline <{self.pipeline_id}> not found.")
        else:
            logger.error(f"unknown pipeline status {json.dumps(res, sort_keys=True, indent=2)}")
