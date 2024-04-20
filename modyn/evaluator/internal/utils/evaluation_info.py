import json
import logging
import pathlib
from typing import Optional

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
        pipeline_id: Optional[int] = None,
        trigger_id: Optional[int] = None,
        num_prefetched_partitions: Optional[int] = None,
        parallel_prefetch_requests: Optional[int] = None,
        selector_address: Optional[str] = None,
    ) -> None: # pragma: no cover
        self.model_id = request.model_id
        self.dataset_id = request.dataset_info.dataset_id
        self.num_dataloaders = request.dataset_info.num_dataloaders

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
        self.tokenizer: Optional[str] = None
        if request.HasField("tokenizer"):
            self.tokenizer = request.tokenizer.value
        else:
            self.tokenizer = None

        self.evaluation_id = evaluation_id
        self.storage_address = storage_address
        self.model_path = model_path

        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.num_prefetched_partitions = num_prefetched_partitions
        self.parallel_prefetch_requests = parallel_prefetch_requests
        self.selector_address = selector_address

        if (
            pipeline_id is not None
            or trigger_id is not None
            or num_prefetched_partitions is not None
            or parallel_prefetch_requests is not None
        ):
            assert (
                pipeline_id is not None
                and trigger_id is not None
                and num_prefetched_partitions is not None
                and parallel_prefetch_requests is not None
                and selector_address is not None
            )
