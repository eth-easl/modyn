import json
import logging
import multiprocessing as mp
import os
import pathlib
from multiprocessing import Manager, Process
from typing import Any, Optional

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.utils import ModelStorageStrategyConfig

# pylint: disable=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import JsonString as SelectorJsonString
from modyn.selector.internal.grpc.generated.selector_pb2 import StrategyConfig
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.grpc.enums import MsgType, PipelineStage, PipelineStatus
from modyn.supervisor.internal.grpc.template_msg import exit_submsg, pipeline_res_msg, pipeline_stage_msg
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import execute_pipeline
from modyn.supervisor.internal.utils import PipelineInfo
from modyn.utils import is_directory_writable, model_available, trigger_available, validate_yaml

logger = logging.getLogger(__name__)


class Supervisor:
    # pylint: disable=too-many-instance-attributes
    # This is a core class and we require the attributes.

    # TODO(#63): Get these from Selector
    supported_strategies: list[str] = [
        "NewDataStrategy",
        "FreshnessSamplingStrategy",
        "CoresetStrategy",
    ]

    supported_evaluation_result_writers: dict = {"json": JsonResultWriter, "tensorboard": TensorboardResultWriter}

    def __init__(
        self,
        modyn_config: dict,
    ) -> None:
        # TODO(#317): redesign tensorboard in the future
        # TODO(#325): validate modyn_config
        self.modyn_config = modyn_config
        self._pipeline_process_dict: dict[int, PipelineInfo] = {}
        self._manager = Manager()
        self._next_pipeline_lock = self._manager.Lock()
        self.grpc = GRPCHandler(self.modyn_config)
        self.init_metadata_db()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if "_manager" in state:
            del state["_manager"]
        else:
            logger.info("'_manager' not found in state")
        return state

    def init_metadata_db(self) -> None:
        with MetadataDatabaseConnection(self.modyn_config) as database:
            database.create_tables()

    def init_cluster_connection(self) -> None:
        logging.info("Setting up connections to cluster components.")
        self.grpc.init_cluster_connection()

        # TODO(#317): seed per pipeline instead of per system in the future
        if "seed" in self.modyn_config:
            self.grpc.seed_selector(self.modyn_config["seed"])

    def validate_pipeline_config_schema(self, pipeline_config: dict) -> bool:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "schema" / "pipeline-schema.yaml"
        )
        valid_yaml, exception = validate_yaml(pipeline_config, schema_path)

        if not valid_yaml:
            logger.error(
                f"Error while validating pipeline configuration file for schema-compliance: {exception.message}"
            )
            logger.error(exception)
            return False

        return True

    def _validate_evaluation_options(self, evaluation_config: dict) -> bool:
        is_valid = True

        dataset_ids = [dataset["dataset_id"] for dataset in evaluation_config["datasets"]]
        if len(set(dataset_ids)) < len(dataset_ids):
            logger.error("Dataset ids must be unique in evaluation")
            is_valid = False

        if "result_writers" in evaluation_config:
            writer_names = set(evaluation_config["result_writers"])
            if diff := writer_names.difference(self.supported_evaluation_result_writers.keys()):
                logger.error(f"Found invalid evaluation result writers: {', '.join(diff)}.")
                is_valid = False

        for dataset in evaluation_config["datasets"]:
            batch_size = dataset["batch_size"]
            if batch_size < 1:
                logger.error(f"Invalid batch size: {batch_size}.")
                is_valid = False

            dataloader_workers = dataset["dataloader_workers"]
            if dataloader_workers < 1:
                logger.error(f"Invalid dataloader worker amount: {dataloader_workers}.")
                is_valid = False

        return is_valid

    # pylint: disable=too-many-branches
    def _validate_training_options(self, training_config: dict) -> bool:
        is_valid = True
        batch_size = training_config["batch_size"]
        dataloader_workers = training_config["dataloader_workers"]
        strategy = training_config["selection_strategy"]["name"]
        initial_model = training_config["initial_model"]

        if training_config["gpus"] != 1:
            logger.error("Currently, only single GPU training is supported.")
            is_valid = False

        if batch_size < 1:
            logger.error(f"Invalid batch size: {batch_size}.")
            is_valid = False

        if dataloader_workers < 1:
            logger.error(f"Invalid dataloader worker amount: {dataloader_workers}.")
            is_valid = False

        if strategy not in Supervisor.supported_strategies:
            logger.error(f"Unsupported strategy: {strategy}. Supported strategies = {Supervisor.supported_strategies}")
            is_valid = False

        if initial_model not in ["random", "pretrained"]:
            logger.error("Only random and pretrained initial models are supported.")
            is_valid = False

        if initial_model == "pretrained":
            if not training_config["use_previous_model"]:
                logger.error(
                    "Cannot have use_previous_model == False and use a pretrained initial model."
                    "Initial model would get lost after first trigger."
                )
                is_valid = False

            if "initial_model_id" not in training_config:
                logger.error("Initial model set to pretrained, but no initial_model_id given")
                is_valid = False

        return is_valid

    def validate_pipeline_config_content(self, pipeline_config: dict) -> bool:
        is_valid = self._validate_training_options(pipeline_config["training"])

        model_id = pipeline_config["model"]["id"]
        if not model_available(model_id):
            logger.error(f"Model {model_id} is not available within Modyn.")
            is_valid = False

        trigger_id = pipeline_config["trigger"]["id"]
        if not trigger_available(trigger_id):
            logger.error(f"Trigger {trigger_id} is not available within Modyn.")
            is_valid = False

        if "evaluation" in pipeline_config:
            is_valid = is_valid and self._validate_evaluation_options(pipeline_config["evaluation"])

        return is_valid

    def validate_pipeline_config(self, pipeline_config: dict) -> bool:
        return self.validate_pipeline_config_schema(pipeline_config) and self.validate_pipeline_config_content(
            pipeline_config
        )

    def dataset_available(self, pipeline_config: dict) -> bool:
        dataset_id = pipeline_config["data"]["dataset_id"]
        available = self.grpc.dataset_available(dataset_id)

        if not available:
            logger.error(f"Dataset {dataset_id} not available at storage.")

        return available

    def validate_system(self, pipeline_config: dict) -> bool:
        dataset_available = self.dataset_available(pipeline_config)
        trainer_server_available = self.grpc.trainer_server_available()
        logger.debug(f"Validate system: dataset {dataset_available}, trainer server {trainer_server_available}")
        return dataset_available and trainer_server_available

    def register_pipeline(self, pipeline_config: dict) -> int:
        """
        Registers a new pipeline in the metadata database.

        Returns:
            The id of the newly created pipeline.
        Throws:
            ValueError if num_workers is not positive.
        """
        num_workers: int = pipeline_config["training"]["dataloader_workers"]
        if num_workers < 0:
            raise ValueError(f"Tried to register training with {num_workers} workers.")

        if "config" in pipeline_config["model"]:
            model_config = json.dumps(pipeline_config["model"]["config"])
        else:
            model_config = "{}"

        model_storage_config = pipeline_config["model_storage"]
        full_model_strategy = ModelStorageStrategyConfig.from_config(
            self.get_model_strategy(model_storage_config["full_model_strategy"])
        )
        incremental_model_strategy_config: Optional[StrategyConfig] = None
        full_model_interval: Optional[int] = None
        if "incremental_model_strategy" in model_storage_config:
            incremental_strategy = model_storage_config["incremental_model_strategy"]
            incremental_model_strategy_config = self.get_model_strategy(incremental_strategy)
            full_model_interval = (
                incremental_strategy["full_model_interval"] if "full_model_interval" in incremental_strategy else None
            )

        incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None
        if incremental_model_strategy_config is not None:
            incremental_model_strategy = ModelStorageStrategyConfig.from_config(incremental_model_strategy_config)

        with self._next_pipeline_lock:
            with MetadataDatabaseConnection(self.modyn_config) as database:
                pipeline_id = database.register_pipeline(
                    num_workers=num_workers,
                    model_class_name=pipeline_config["model"]["id"],
                    model_config=model_config,
                    amp=pipeline_config["training"]["amp"] if "amp" in pipeline_config["training"] else False,
                    selection_strategy=json.dumps(pipeline_config["training"]["selection_strategy"]),
                    full_model_strategy=full_model_strategy,
                    incremental_model_strategy=incremental_model_strategy,
                    full_model_interval=full_model_interval,
                )

        return pipeline_id

    @staticmethod
    def get_model_strategy(strategy_config: dict) -> StrategyConfig:
        return StrategyConfig(
            name=strategy_config["name"],
            zip=strategy_config["zip"] if "zip" in strategy_config else None,
            zip_algorithm=strategy_config["zip_algorithm"] if "zip_algorithm" in strategy_config else None,
            config=SelectorJsonString(value=json.dumps(strategy_config["config"]))
            if "config" in strategy_config
            else None,
        )

    def unregister_pipeline(self, pipeline_id: int) -> None:
        # TODO(#64,#124,#317): Implement.
        pass

    def start_pipeline(
        self,
        pipeline_config: dict,
        eval_directory: str,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> dict:
        if not self.validate_pipeline_config(pipeline_config):
            return pipeline_res_msg(exception="Invalid pipeline configuration")

        if not is_directory_writable(pathlib.Path(eval_directory)):
            return pipeline_res_msg(exception="No permission to write to the evaluation results directory.")

        if not self.validate_system(pipeline_config):
            return pipeline_res_msg(exception="Invalid system configuration")

        try:
            exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
            pipeline_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object
            training_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object
            eval_status_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

            start_timestamp = self.grpc.get_time_at_storage()
            pipeline_id = self.register_pipeline(pipeline_config)
            logger.info(f"Pipeline {pipeline_id} registered, start executing.")
        except Exception:  # pylint: disable=broad-except
            return pipeline_res_msg(exception="Failed to register pipeline")

        try:
            process = Process(
                target=execute_pipeline,
                args=(
                    start_timestamp,
                    pipeline_id,
                    self.modyn_config,
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
                process, exception_queue, pipeline_status_queue, training_status_queue, eval_status_queue
            )
            return pipeline_res_msg(pipeline_id=pipeline_id)
        except Exception:  # pylint: disable=broad-except
            return pipeline_res_msg(pipeline_id=pipeline_id, exception="Failed to execute pipeline")

    def get_all_msgs_from_queue(self, p_info: PipelineInfo, queue_name: str):
        msgs = []

        pipeline_stage = p_info.get_msg_from_queue(queue_name)
        while pipeline_stage is not None:
            msgs.append(pipeline_stage)
            pipeline_stage = p_info.get_msg_from_queue(queue_name)

        return msgs

    def get_pipeline_status(self, pipeline_id: int) -> dict[str, Any]:
        ret: dict[str, Any] = {}

        if pipeline_id not in self._pipeline_process_dict:
            ret["status"] = str(PipelineStatus.NOTFOUND)
            return ret

        p_info = self._pipeline_process_dict[pipeline_id]

        ret["pipeline_stage"] = self.get_all_msgs_from_queue(p_info, "pipeline_status_queue")
        ret["training_status"] = self.get_all_msgs_from_queue(p_info, "training_status_queue")
        ret["eval_status"] = self.get_all_msgs_from_queue(p_info, "eval_status_queue")

        if p_info.process_handler.is_alive():
            ret["status"] = str(PipelineStatus.RUNNING)
        else:
            ret["status"] = str(PipelineStatus.EXIT)

            msg: dict[str, Any] = pipeline_stage_msg(
                stage=PipelineStage.EXIT,
                msg_type=MsgType.EXIT,
                submsg=exit_submsg(p_info.process_handler.exitcode, p_info.check_for_exception()),
            )

            ret["pipeline_stage"].append(msg)

        return ret
