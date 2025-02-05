import json
import logging
import pathlib
from typing import Any

# pylint: disable=no-name-in-module
# from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import StartTrainingtuning_info
#from modyn.utils.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class TuningInfo:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        tuning_info: Any,
        evaluation_id: int,
        amp: bool,
        storage_address: str,
        selector_address: str,
        offline_dataset_path: str,
        log_file_path: pathlib.Path,
    ) -> None:
        self.pipeline_id = tuning_info.pipeline_id
        self.trigger_id = tuning_info.trigger_id
        self.evaluation_id = evaluation_id
        self.num_prefetched_partitions = tuning_info.num_prefetched_partitions
        self.parallel_prefetch_tuning_infos = tuning_info.parallel_prefetch_tuning_infos

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
        self.selector_address = selector_address
        self.storage_address = storage_address

       # model_module = dynamic_module_import("modyn.models")

        self.use_pretrained_model = tuning_info.use_pretrained_model
        self.load_optimizer_state = tuning_info.load_optimizer_state
        self.log_file_path = log_file_path
        self.shuffle = tuning_info.shuffle
        self.enable_accurate_gpu_measurements = tuning_info.enable_accurate_gpu_measurements
        self.generative = tuning_info.generative
        self.no_labels = tuning_info.no_labels
        self.steps = tuning_info.steps

        self.batch_size = tuning_info.batch_size
        self.drop_last_batch = tuning_info.drop_last_batch
        self.torch_criterion = tuning_info.torch_criterion
        self.amp = amp

        self.lr_scheduler = json.loads(tuning_info.lr_scheduler.value)

        self.checkpoint_path = pathlib.Path(tuning_info.checkpoint_info.checkpoint_path)
        self.checkpoint_interval = tuning_info.checkpoint_info.checkpoint_interval
        self.record_loss_every = tuning_info.record_loss_every
        self.seed: int | None = tuning_info.seed if tuning_info.HasField("seed") else None
        self.tokenizer: str | None = tuning_info.tokenizer.value if tuning_info.HasField("tokenizer") else None

        self.offline_dataset_path = offline_dataset_path
