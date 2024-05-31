from typing import Optional

import grpc

# pylint: disable-next=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    GetNumberOfPartitionsRequest,
    GetSamplesRequest,
    NumberOfPartitionsResponse,
    UsesWeightsRequest,
    UsesWeightsResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource
from modyn.utils import MAX_MESSAGE_SIZE, flatten, grpc_connection_established


class SelectorKeySource(AbstractKeySource):
    def __init__(self, pipeline_id: int, trigger_id: int, selector_address: str) -> None:
        super().__init__(pipeline_id, trigger_id)

        self._selector_address = selector_address
        self._selectorstub = None  # connection is made when the pytorch worker is started
        self._uses_weights: Optional[bool] = None  # get via gRPC, so unavailable if the connection is not yet made.

    def get_keys_and_weights(self, worker_id: int, partition_id: int, shuffle: bool) -> tuple[list[int], Optional[list[float]]]:
        assert self._selectorstub is not None
        assert self._uses_weights is not None

        req = GetSamplesRequest(
            pipeline_id=self._pipeline_id, trigger_id=self._trigger_id, worker_id=worker_id, partition_id=partition_id, shuffle=shuffle
        )

        if self._uses_weights:
            return self._get_both_keys_and_weights(req)
        return self._get_just_keys(req)

    def _get_just_keys(self, req: GetSamplesRequest) -> tuple[list[int], Optional[list[float]]]:
        assert self._selectorstub is not None
        assert not self._uses_weights

        keys = flatten(
            [response.training_samples_subset for response in self._selectorstub.get_sample_keys_and_weights(req)]
        )
        weights = None

        return keys, weights

    def _get_both_keys_and_weights(self, req: GetSamplesRequest) -> tuple[list[int], list[float]]:
        assert self._selectorstub is not None
        assert self._uses_weights

        keys_and_weights = [
            (response.training_samples_subset, response.training_samples_weights)
            for response in self._selectorstub.get_sample_keys_and_weights(req)
        ]

        keys = flatten([element[0] for element in keys_and_weights])
        weights = flatten([element[1] for element in keys_and_weights])

        return keys, weights

    def get_num_data_partitions(self) -> int:
        assert self._selectorstub is not None

        num_partitions_request = GetNumberOfPartitionsRequest(
            pipeline_id=self._pipeline_id,
            trigger_id=self._trigger_id,
        )

        response: NumberOfPartitionsResponse = self._selectorstub.get_number_of_partitions(num_partitions_request)
        return response.num_partitions

    def uses_weights(self) -> bool:
        assert self._selectorstub is not None

        if self._uses_weights is not None:
            # we can cache the response
            return self._uses_weights

        req = UsesWeightsRequest(pipeline_id=self._pipeline_id)
        response: UsesWeightsResponse = self._selectorstub.uses_weights(req)
        return response.uses_weights

    def init_worker(self) -> None:
        self._selectorstub = self._connect_to_selector()
        self._uses_weights = self.uses_weights()

    def _connect_to_selector(self) -> SelectorStub:
        selector_channel = grpc.insecure_channel(
            self._selector_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        if not grpc_connection_established(selector_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to selector at address {self._selector_address}."
            )
        return SelectorStub(selector_channel)

    def end_of_trigger_cleaning(self) -> None:
        pass
