from __future__ import annotations

import json
import logging
import multiprocessing as mp
import pathlib
from multiprocessing import Manager, Process
from typing import Any, Optional

from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import ModelStrategy, ModynPipelineConfig, ResultWriterType
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig

# pylint: disable=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import JsonString as SelectorJsonString
from modyn.selector.internal.grpc.generated.selector_pb2 import StrategyConfig
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.evaluation_result_writer.abstract_evaluation_result_writer import (
    AbstractEvaluationResultWriter,
)
from modyn.supervisor.internal.grpc.enums import MsgType, PipelineStage, PipelineStatus
from modyn.supervisor.internal.grpc.template_msg import exit_submsg, pipeline_res_msg, pipeline_stage_msg
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import execute_pipeline
from modyn.supervisor.internal.utils import PipelineInfo
from modyn.utils import is_directory_writable, model_available, trigger_available
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class Supervisor:
    # pylint: disable=too-many-instance-attributes
    # This is a core class and we require the attributes.

    supported_evaluation_result_writers: dict[ResultWriterType, type[AbstractEvaluationResultWriter]] = {
        "json": JsonResultWriter,
        "tensorboard": TensorboardResultWriter,
    }

    def __init__(self, modyn_config: ModynConfig) -> None:
        # TODO(#317): redesign tensorboard in the future
        self.modyn_config = modyn_config
        self._pipeline_process_dict: dict[int, PipelineInfo] = {}
        self._manager = Manager()
        self._next_pipeline_lock = self._manager.Lock()
        self.grpc = GRPCHandler(self.modyn_config.model_dump(by_alias=True))
        self.init_metadata_db()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if "_manager" in state:
            del state["_manager"]
        else:
            logger.info("'_manager' not found in state")
        return state

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                  INITIALIZATION                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    # -------------------------------------------------- Connections ------------------------------------------------- #

    def init_metadata_db(self) -> None:
        with MetadataDatabaseConnection(self.modyn_config.model_dump(by_alias=True)) as database:
            database.create_tables()

    def init_cluster_connection(self) -> None:
        logging.info("Setting up connections to cluster components.")
        self.grpc.init_cluster_connection()

        # TODO(#317): seed per pipeline instead of per system in the future
        if "seed" in self.modyn_config:
            self.grpc.seed_selector(self.modyn_config["seed"])

    @classmethod
    def validate_pipeline_config_content(cls, pipeline_config: ModynPipelineConfig) -> bool:
        is_valid = True
        model_id = pipeline_config.modyn_model.id
        if not model_available(model_id):
            logger.error(f"Model {model_id} is not available within Modyn.")
            is_valid = False

        trigger_id = pipeline_config.trigger.id
        if not trigger_available(trigger_id):
            logger.error(f"Trigger {trigger_id} is not available within Modyn.")
            is_valid = False

        return is_valid

    @classmethod
    def validate_pipeline_config(cls, pipeline_config: dict) -> ModynPipelineConfig | None:
        """Validates the pipeline configuration.

        Args:
            pipeline_config: The pipeline configuration.

        Returns:
            ModynPipelineConfig if the pipeline configuration is valid, or None if it is invalid.
        """
        try:
            pipeline_config_model = ModynPipelineConfig.model_validate(pipeline_config)
            if not cls.validate_pipeline_config_content(pipeline_config_model):
                return None
            return pipeline_config_model
        except ValidationError:
            return None

    @staticmethod
    def get_model_strategy(strategy_config: ModelStrategy) -> StrategyConfig:
        return StrategyConfig(
            name=strategy_config.name,
            zip=strategy_config.zip,
            zip_algorithm=strategy_config.zip_algorithm,
            config=(SelectorJsonString(value=json.dumps(strategy_config.config)) if strategy_config.config else None),
        )

    def dataset_available(self, pipeline_config: ModynPipelineConfig) -> bool:
        available = self.grpc.dataset_available(pipeline_config.data.dataset_id)
        if not available:
            logger.error(f"Dataset {pipeline_config.data.dataset_id} not available at storage.")
        return available

    def validate_system(self, pipeline_config: ModynPipelineConfig) -> bool:
        dataset_available = self.dataset_available(pipeline_config)
        trainer_server_available = self.grpc.trainer_server_available()
        logger.debug(f"Validate system: dataset {dataset_available}, trainer server {trainer_server_available}")
        return dataset_available and trainer_server_available

    # ----------------------------------------------- Setup & teardown ----------------------------------------------- #

    def register_pipeline(self, pipeline_config: ModynPipelineConfig) -> int:
        """
        Registers a new pipeline in the metadata database.

        Returns:
            The id of the newly created pipeline.
        Throws:
            ValueError if num_workers is not positive.
        """
        num_workers = pipeline_config.training.dataloader_workers
        if num_workers < 0:
            raise ValueError(f"Tried to register training with {num_workers} workers.")

        model_config_str = json.dumps(pipeline_config.modyn_model.config)

        model_storage_config = pipeline_config.modyn_model_storage
        full_model_strategy = ModelStorageStrategyConfig.from_config(
            self.get_model_strategy(model_storage_config.full_model_strategy)
        )
        incremental_model_strategy_config: Optional[StrategyConfig] = None
        full_model_interval: Optional[int] = None
        incremental_strategy = model_storage_config.incremental_model_strategy
        if incremental_strategy:
            incremental_model_strategy_config = self.get_model_strategy(incremental_strategy)
            full_model_interval = incremental_strategy.full_model_interval

        incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None
        if incremental_model_strategy_config:
            incremental_model_strategy = ModelStorageStrategyConfig.from_config(incremental_model_strategy_config)

        with self._next_pipeline_lock:
            with MetadataDatabaseConnection(self.modyn_config.model_dump(by_alias=True)) as database:
                pipeline_id = database.register_pipeline(
                    num_workers=num_workers,
                    model_class_name=pipeline_config.modyn_model.id,
                    model_config=model_config_str,
                    amp=pipeline_config.training.amp,
                    selection_strategy=pipeline_config.training.selection_strategy.model_dump_json(),
                    full_model_strategy=full_model_strategy,
                    incremental_model_strategy=incremental_model_strategy,
                    full_model_interval=full_model_interval,
                )

        return pipeline_id

    def unregister_pipeline(self, pipeline_id: int) -> None:
        # TODO(#64,#124,#317): Implement.
        pass

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                   PIPELINE RUN                                                   #
    # ---------------------------------------------------------------------------------------------------------------- #

    def start_pipeline(
        self,
        pipeline_config: dict,
        eval_directory: str,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> dict:
        pipeline_config_model = self.validate_pipeline_config(pipeline_config)
        if not pipeline_config_model:
            return pipeline_res_msg(exception="Invalid pipeline configuration")

        if not is_directory_writable(pathlib.Path(eval_directory)):
            return pipeline_res_msg(exception="No permission to write to the evaluation results directory.")

        if not self.validate_system(pipeline_config_model):
            return pipeline_res_msg(exception="Invalid system configuration")

        try:
            exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
            pipeline_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object
            training_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object
            eval_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

            start_timestamp = self.grpc.get_time_at_storage()
            pipeline_id = self.register_pipeline(pipeline_config_model)
            logger.info(f"Pipeline {pipeline_id} registered, start executing.")
        except Exception:  # pylint: disable=broad-except
            return pipeline_res_msg(exception="Failed to register pipeline")

        try:
            process = Process(
                target=execute_pipeline,
                args=(
                    start_timestamp,
                    pipeline_id,
                    self.modyn_config.model_dump(by_alias=True),
                    pipeline_config,
                    eval_directory,
                    self.supported_evaluation_result_writers,
                    exception_queue,
                    pipeline_status_queue,
                    training_status_queue,
                    eval_status_queue,
                    start_replay_at,
                    stop_replay_at,
                    maximum_triggers,
                ),
            )
            process.start()
            self._pipeline_process_dict[pipeline_id] = PipelineInfo(
                process,
                exception_queue,
                pipeline_status_queue,
                training_status_queue,
                eval_status_queue,
            )
            return pipeline_res_msg(pipeline_id=pipeline_id)
        except Exception:  # pylint: disable=broad-except
            return pipeline_res_msg(pipeline_id=pipeline_id, exception="Failed to execute pipeline")

    # ---------------------------------------------------- Helpers --------------------------------------------------- #

    def get_pipeline_status(self, pipeline_id: int) -> dict[str, Any]:
        ret: dict[str, Any] = {}

        if pipeline_id not in self._pipeline_process_dict:
            ret["status"] = str(PipelineStatus.NOTFOUND)
            return ret

        p_info = self._pipeline_process_dict[pipeline_id]

        ret["pipeline_stage"] = p_info.get_all_msgs_from_queue("pipeline_status_queue")
        ret["training_status"] = p_info.get_all_msgs_from_queue("training_status_queue")
        ret["eval_status"] = p_info.get_all_msgs_from_queue("eval_status_queue")

        if p_info.process_handler.is_alive():
            ret["status"] = str(PipelineStatus.RUNNING)
        else:
            ret["status"] = str(PipelineStatus.EXIT)
            p_info.process_handler.join()

            msg: dict[str, Any] = pipeline_stage_msg(
                stage=PipelineStage.EXIT,
                msg_type=MsgType.EXIT,
                submsg=exit_submsg(p_info.process_handler.exitcode or 0, p_info.check_for_exception()),
            )

            ret["pipeline_stage"].append(msg)

        return ret
