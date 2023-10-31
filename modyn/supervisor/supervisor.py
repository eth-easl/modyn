import logging
import os
import pathlib
from multiprocessing import Lock, Manager, Process
from typing import Optional

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.pipeline_executor import PipelineExecutor
from modyn.utils import model_available, trigger_available, validate_yaml

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
        self.modyn_config = modyn_config
        self._manager = Manager()
        self._next_pipeline_lock = self._manager.Lock()
        self.grpc = GRPCHandler(self.modyn_config)

        logging.info("Setting up connections to cluster components.")
        self.init_metadata_db()

        # TODO(#317): seed per pipeline instead of per system
        if "seed" in self.modyn_config:
            self.grpc.seed_selector(self.modyn_config["seed"])

        # TODO(#317): redesign tensorboard. ignore it for now
        # if "tensorboard" in self.modyn_config:
        #     port = self.modyn_config["tensorboard"]["port"]
        #     self._run_tensorboard(port)
        #     logger.info(f"Starting up tensorboard on port {port}.")

    # TODO(#317): what is it?
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_manager"]
        return state

    def init_metadata_db(self) -> None:
        with MetadataDatabaseConnection(self.modyn_config) as database:
            database.create_tables()

    def validate_pipeline_config_schema(self, pipeline_config: dict) -> bool:
        schema_path = (
            pathlib.Path(os.path.abspath(__file__)).parent.parent / "config" / "schema" / "pipeline-schema.yaml"
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

        if training_config["initial_pass"]["activated"]:
            reference = training_config["initial_pass"]["reference"]
            if reference not in ("amount", "timestamp"):
                logger.error(f"Invalid reference for initial pass: {reference} (valid are 'amount' or 'timestamp')")
                is_valid = False

            if reference == "amount":
                amount = training_config["initial_pass"]["amount"]
                if float(amount) > 1.0 or float(amount) < 0:
                    logger.error(f"Invalid initial pass amount: {amount}")
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

    # def _run_tensorboard(self, port: str) -> None:
    #     logging.getLogger("tensorboard").setLevel(logging.ERROR)
    #     logging.getLogger("MARKDOWN").setLevel(logging.ERROR)

    #     tensorboard = program.TensorBoard()
    #     tensorboard.configure(
    #         argv=[
    #             None,
    #             "--logdir",
    #             str(self.eval_directory),
    #             "--bind_all",
    #             "--port",
    #             port,
    #             "--window_title",
    #             "Modyn TensorBoard",
    #         ]
    #     )
    #     tensorboard.launch()
    #     logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def pipeline(
        self,
        next_pipeline_lock: Lock,
        modyn_config: dict,
        pipeline_config: dict,
        eval_directory: pathlib.Path,
        supervisor_supported_eval_result_writers: dict,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> None:
        pipeline = PipelineExecutor(
            next_pipeline_lock,
            modyn_config,
            pipeline_config,
            eval_directory,
            supervisor_supported_eval_result_writers,
            start_replay_at,
            stop_replay_at,
            maximum_triggers,
        )
        pipeline.register_and_execute()
        logger.info("Pipeline done.")

    def start_pipeline(
        self,
        pipeline_config: dict,
        eval_directory: pathlib.Path,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> int:
        if not self.validate_pipeline_config(pipeline_config):
            raise ValueError("Invalid pipeline configuration")

        # TODO(#317): start a process. pool?
        # TODO(#317): return pipeline id or something else?
        process = Process(
            target=self.pipeline,
            args=(
                self._next_pipeline_lock,
                self.modyn_config,
                pipeline_config,
                eval_directory,
                self.supported_evaluation_result_writers,
                start_replay_at,
                stop_replay_at,
                maximum_triggers,
            ),
        )
        process.start()
