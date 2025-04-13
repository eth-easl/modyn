import json
import logging
import pathlib
from typing import Any

# pylint: disable=no-name-in-module
# from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import StartTrainingtuning_info

logger = logging.getLogger(__name__)


class TuningInfo:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        tuning_info: Any,
        evaluation_id: int,
        offline_dataset_path: str,
        log_file_path: pathlib.Path,
    ) -> None:
        self.pipeline_id = tuning_info.pipeline_id

        self.evaluation_id = evaluation_id
        self.device = tuning_info.device
        self.dataset_id = tuning_info.data_info.dataset_id
        self.num_dataloaders = tuning_info.data_info.num_dataloaders
        self.epochs = tuning_info.epochs
        self.num_samples_to_pass = tuning_info.num_samples_to_pass

        self.torch_optimizers_configuration = json.loads(tuning_info.torch_optimizers_configuration.value)
        self.criterion_dict = json.loads(tuning_info.criterion_parameters.value)
        self.grad_scaler_configuration = json.loads(tuning_info.grad_scaler_configuration.value)

        self.transform_list = list(tuning_info.transform_list)
        self.bytes_parser = tuning_info.bytes_parser.value
        self.label_transformer = tuning_info.label_transformer.value

        self.log_file_path = log_file_path
        self.shuffle = tuning_info.shuffle
        self.enable_accurate_gpu_measurements = tuning_info.enable_accurate_gpu_measurements
        self.generative = tuning_info.generative
        self.steps = tuning_info.steps
        self.batch_size = tuning_info.batch_size
        self.drop_last_batch = tuning_info.drop_last_batch
        self.torch_criterion = tuning_info.torch_criterion
        self.amp = tuning_info.amp

        self.lr_scheduler = json.loads(tuning_info.lr_scheduler.value)

        self.record_loss_every = tuning_info.record_loss_every
        self.seed: int | None = tuning_info.seed if tuning_info.seed is not None else None
        self.tokenizer: str | None = tuning_info.tokenizer.value if tuning_info.tokenizer is not None else None
        self.bytes_parser_target: str | None = (
            tuning_info.bytes_parser_target.value if tuning_info.bytes_parser_target is not None else None
        )
        self.serialized_transforms_target: list[str] | None = (
            list(tuning_info.serialized_transforms_target)
            if tuning_info.serialized_transforms_target is not None
            else None
        )
        self.offline_dataset_path = offline_dataset_path
