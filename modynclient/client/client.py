import logging
import pathlib
from typing import Optional
from modynclient.client.internal.grpc_handler import GRPCHandler

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
        self.pipeline_id = self.grpc.start_pipeline(
            self.pipeline_config, 
            self.eval_directory, 
            self.start_replay_at, 
            self.stop_replay_at, 
            self.maximum_triggers
        )
    

    def poll_pipeline_status(self) -> None:
        pass
