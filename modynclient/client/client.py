import json
import logging
import pathlib
import time
from typing import Optional

import enlighten
from modynclient.client.internal.grpc_handler import GRPCHandler
from modynclient.client.internal.utils import TrainingStatusTracker, EvaluationStatusTracker

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

    def monitor_pipeline_progress(self, msg: dict) -> None:
        self.status_bar.update(demo=msg["stage"])
        if msg["stage"] == "handle new data":
            if "new_data_len" in msg:
                self.pbar = self.progress_mgr.counter(
                    total=msg["new_data_len"], desc=f"[Pipeline {self.pipeline_id}] Processing New Samples", unit="samples"
                )
            else:
                assert self.pbar is not None
                self.pbar.update(msg["batch_size"])
        elif msg["stage"] == "new data handled":
            assert self.pbar is not None
            self.pbar.clear(flush=True)
            self.pbar.close(clear=True)

    def monitor_training_progress(self, msg: dict) -> None:
        if msg["stage"] == "wait for training":
            if msg["action"] == "create_tracker":
                self.training_status_tracker = TrainingStatusTracker(
                    self.progress_mgr, msg["training_id"], msg["total_samples"],  msg["status_bar_scale"]
                )
            elif msg["action"] == "progress_counter":
                self.training_status_tracker.progress_counter(msg["samples_seen"], msg["downsampling_samples_seen"], msg["is_training"])
            elif msg["action"] == "close_counter":
                self.training_status_tracker.close_counter()
            elif msg["action"] == "update_status_bar":
                logger.info(f"Pipeline <{self.pipeline_id}> {msg['demo']} (id = {msg['training_id']})")
                self.status_bar.update(demo=msg["demo"])


    def poll_pipeline_status(self) -> None:
        res = self.grpc.get_pipeline_status(self.pipeline_id)

        while res["status"] == "running":
            # print(json.dumps(res, sort_keys=True, indent=2))
            if "pipeline_status_detail" in res:
                self.monitor_pipeline_progress(res["pipeline_status_detail"])

            if "training_status_detail" in res:
                self.monitor_training_progress(res["training_status_detail"])
                
            time.sleep(POLL_TIMEOUT)

            res = self.grpc.get_pipeline_status(self.pipeline_id)
            
        if res["status"] == "exit":
            if res["pipeline_status_detail"]["exitcode"] == 0:
                logger.info(f"Pipeline <{self.pipeline_id}> finished successfully 🚀.")
            else:
                logger.info(f"Pipeline <{self.pipeline_id}> exited with error {res['pipeline_status_detail']['exception']}")
        elif res["status"] == "not found":
            logger.info(f"Pipeline <{self.pipeline_id}> not found.")
        else:
            print(json.dumps(res, sort_keys=True, indent=2))