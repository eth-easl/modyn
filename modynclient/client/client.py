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

    def start_pipeline(self) -> None:
        logger.info(f"model id: {self.pipeline_config['model']['id']}, maximum_triggers: {self.maximum_triggers}")
        self.pipeline_id = self.grpc.start_pipeline(
            self.pipeline_config,
            self.eval_directory,
            self.start_replay_at,
            self.stop_replay_at,
            self.maximum_triggers
        )
        logger.info(f"Pipeline <{self.pipeline_id}> started.")

    def _monitor_pipeline_progress(self, msg: dict) -> None:
        if msg["log"]:
            logger.info(msg)

        msg_type = msg['msg_type']
        submsg = msg['msg']

        demo = f"Pipeline <{self.pipeline_id}> {msg['stage']}"

        if msg_type == "general":
            pass
        elif msg_type == "counter":
            if submsg["action"] == "create":
                self.pbar = self.progress_mgr.counter(
                    total=submsg["new_data_len"],
                    desc=f"[Pipeline {self.pipeline_id}] Processing New Samples",
                    unit="samples"
                )
            elif submsg["action"] == "update":
                assert self.pbar is not None
                self.pbar.update(submsg["batch_size"])
            elif submsg["action"] == "close":
                assert self.pbar is not None
                self.pbar.clear(flush=True)
                self.pbar.close(clear=True)
        else:
            demo += f" ({msg_type} = {submsg['id']})"

        self.status_bar.update(demo=demo)

    def _monitor_training_progress(self, msg: dict) -> None:
        if msg["stage"] == "wait for training":
            if msg["action"] == "create_tracker":
                self.training_status_tracker = TrainingStatusTracker(
                    self.progress_mgr, msg["training_id"], msg["total_samples"],  msg["status_bar_scale"]
                )
            elif msg["action"] == "progress_counter":
                self.training_status_tracker.progress_counter(
                    msg["samples_seen"], msg["downsampling_samples_seen"], msg["is_training"]
                )
            elif msg["action"] == "close_counter":
                self.training_status_tracker.close_counter()
        elif msg["stage"] == "wait for evaluation":
            if msg["action"] == "create_tracker":
                self.evaluations[msg["evaluation_id"]] = EvaluationStatusTracker(
                    msg["dataset_id"], msg["dataset_size"]
                )
            elif msg["action"] == "create_counter":
                self.evaluations[msg["evaluation_id"]].create_counter(
                    self.progress_mgr, msg["training_id"], msg["evaluation_id"]
                )
            elif msg["action"] == "progress_counter":
                self.evaluations[msg["evaluation_id"]].progress_counter(msg["total_samples_seen"])
            elif msg["action"] == "end_counter":
                self.evaluations[msg["evaluation_id"]].end_counter(msg["error"])

    def poll_pipeline_status(self) -> None:
        res = self.grpc.get_pipeline_status(self.pipeline_id)

        while res["status"] == "running":
            # print(json.dumps(res, sort_keys=True, indent=2))
            if "pipeline_status_detail" in res:
                self._monitor_pipeline_progress(res["pipeline_status_detail"])

            if "training_status_detail" in res:
                self._monitor_training_progress(res["training_status_detail"])

            time.sleep(POLL_TIMEOUT)

            res = self.grpc.get_pipeline_status(self.pipeline_id)

        if res["status"] == "exit":
            if res["pipeline_status_detail"]["exitcode"] == 0:
                logger.info(f"Pipeline <{self.pipeline_id}> finished successfully.")
            else:
                logger.info(
                    f"Pipeline <{self.pipeline_id}> exited with error {res['pipeline_status_detail']['exception']}"
                )
        elif res["status"] == "not found":
            logger.info(f"Pipeline <{self.pipeline_id}> not found.")
        else:
            print(json.dumps(res, sort_keys=True, indent=2))
