import json
import logging
import pathlib
from typing import Optional

# pylint: disable=no-name-in-module
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import StartTrainingRequest
from modyn.utils.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class TrainingInfo:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        request: StartTrainingRequest,
        training_id: int,
        storage_address: str,
        selector_address: str,
        offline_dataset_path: str,
        final_checkpoint_path: pathlib.Path,
        pretrained_model_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.pipeline_id = request.pipeline_id
        self.trigger_id = request.trigger_id
        self.training_id = training_id

        self.dataset_id = request.data_info.dataset_id
        self.num_dataloaders = request.data_info.num_dataloaders
        self.epochs_per_trigger = request.epochs_per_trigger

        self.torch_optimizers_configuration = json.loads(request.torch_optimizers_configuration.value)
        self.model_configuration_dict = json.loads(request.model_configuration.value)
        self.criterion_dict = json.loads(request.criterion_parameters.value)
        self.grad_scaler_configuration = json.loads(request.grad_scaler_configuration.value)

        self.transform_list = list(request.transform_list)
        self.bytes_parser = request.bytes_parser.value
        self.label_transformer = request.label_transformer.value

        self.model_id = request.model_id
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, self.model_id)

        self.use_pretrained_model = request.use_pretrained_model
        self.load_optimizer_state = request.load_optimizer_state
        self.pretrained_model_path = pretrained_model_path

        if self.use_pretrained_model:
            assert self.pretrained_model_path, "Inconsistent pretrained model configuration"

        self.batch_size = request.batch_size
        self.torch_criterion = request.torch_criterion
        self.amp = request.amp

        self.lr_scheduler = json.loads(request.lr_scheduler.value)

        self.checkpoint_path = pathlib.Path(request.checkpoint_info.checkpoint_path)
        self.checkpoint_interval = request.checkpoint_info.checkpoint_interval

        self.storage_address = storage_address
        self.selector_address = selector_address

        self.final_checkpoint_path = final_checkpoint_path

        self.seed: Optional[int]
        if request.HasField("seed"):
            self.seed = request.seed
        else:
            self.seed = None

        self.tokenizer: Optional[str]
        if request.HasField("tokenizer"):
            self.tokenizer = request.tokenizer.value
        else:
            self.tokenizer = None

        self.offline_dataset_path = offline_dataset_path
