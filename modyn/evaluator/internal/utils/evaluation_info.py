import json
import logging
import pathlib

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluateModelRequest
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class EvaluationInfo:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        request: EvaluateModelRequest,
        evaluation_id: int,
        model_class_name: str,
        model_config: str,
        amp: bool,
        storage_address: str,
        metrics: list[AbstractEvaluationMetric],
        model_path: pathlib.Path,
    ) -> None:  # pragma: no cover
        self.model_id = request.model_id
        self.dataset_id = request.dataset_info.dataset_id
        self.num_dataloaders = request.dataset_info.num_dataloaders
        self.start_timestamp = (
            request.dataset_info.start_timestamp if request.dataset_info.HasField("start_timestamp") else None
        )
        self.end_timestamp = (
            request.dataset_info.end_timestamp if request.dataset_info.HasField("end_timestamp") else None
        )
        self.device = request.device
        self.amp = amp
        self.batch_size = request.batch_size
        self.metrics = metrics

        self.model_class_name = model_class_name
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, self.model_class_name)
        self.model_configuration_dict = json.loads(model_config)

        self.transform_list = list(request.transform_list)
        self.bytes_parser = request.bytes_parser.value
        self.label_transformer = request.label_transformer.value
        self.tokenizer = request.tokenizer.value if request.HasField("tokenizer") else None

        self.evaluation_id = evaluation_id
        self.storage_address = storage_address
        self.model_path = model_path
