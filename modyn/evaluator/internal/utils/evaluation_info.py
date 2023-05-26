import json
import logging
import pathlib

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelRequest
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class EvaluationInfo:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        request: EvaluateModelRequest,
        evaluation_id: int,
        storage_address: str,
        model_path: pathlib.Path,
    ) -> None:
        self.trained_model_id = request.trained_model_id
        self.dataset_id = request.data_info.dataset_id
        self.num_dataloaders = request.data_info.num_dataloaders

        self.device = request.device
        self.amp = request.amp
        self.batch_size = request.batch_size
        self.evaluation_layer = request.evaluation_layer
        self.metrics = list(request.metrics)

        self.model_id = request.model_id
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, self.model_id)
        self.model_configuration_dict = json.loads(request.model_configuration.value)

        self.transform_list = list(request.transform_list)
        self.bytes_parser = request.bytes_parser.value
        self.label_transformer = request.label_transformer.value

        self.evaluation_id = evaluation_id
        self.storage_address = storage_address
        self.model_path = model_path
