import json
import logging
import pathlib
import time
from typing import Optional
from modynclient.client.internal.grpc_handler import GRPCHandler

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


    def start_pipeline(self) -> None:
        logger.info(f"model id: {self.pipeline_config['model']['id']}, max triggers: {self.maximum_triggers}")
        self.pipeline_id = self.grpc.start_pipeline(
            self.pipeline_config, 
            self.eval_directory, 
            self.start_replay_at, 
            self.stop_replay_at, 
            self.maximum_triggers
        )
        logger.info(f"Pipeline started: <{self.pipeline_id}>")
    

    def poll_pipeline_status(self) -> None:
        res = self.grpc.get_pipeline_status(self.pipeline_id)

        while res["status"] == "running":
            print(json.dumps(res, sort_keys=True, indent=2))
            res = self.grpc.get_pipeline_status(self.pipeline_id)
            time.sleep(POLL_TIMEOUT)
            
        print(json.dumps(res, sort_keys=True, indent=2))