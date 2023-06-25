import json

import grpc
from integrationtests.utils import get_modyn_config
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    GetNumberOfPartitionsRequest,
    GetSamplesRequest,
    JsonString,
    RegisterPipelineRequest,
    SamplesResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.utils import grpc_connection_established

# TODO(54): Write more integration tests for different strategies.


def connect_to_selector_servicer() -> grpc.Channel:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

    return selector_channel


def test_newdata() -> None:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test the NewData strategy for finetuning on the new data, i.e., we reset without limit
    # We also enforce high partitioning (maximum_keys_in_memory == 2) to ensure that works

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 2,
        "config": {"limit": -1, "reset_after_trigger": True},
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[0, 1, 2],
            timestamps=[1, 2, 3],
            labels=[1, 0, 1],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[3, 4, 5],
            timestamps=[4, 5, 6],
            labels=[0, 0, 1],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 3, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) == 1
        assert len(worker_2_samples) == 1

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(6)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 6

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[6, 7, 8],
            timestamps=[7, 8, 9],
            labels=[1, 0, 1],
        )
    )

    next_trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[9, 10, 11],
            timestamps=[10, 11, 12],
            labels=[0, 0, 1],
        )
    ).trigger_id

    assert next_trigger_id > trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=next_trigger_id)
    ).num_partitions

    assert number_of_partitions == 3, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=0, partition_id=partition
                )
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=1, partition_id=partition
                )
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) == 1
        assert len(worker_2_samples) == 1

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(6, 12)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 6


def test_abstract_downsampler(reset_after_trigger) -> None:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    # sampling every datapoint
    strategy_config = {
        "name": "LossDownsamplingStrategy",
        "maximum_keys_in_memory": 50000,
        "config": {
            "limit": -1,
            "reset_after_trigger": reset_after_trigger,
            "presampling_ratio": 20,
            "downsampled_batch_size": 10,
            "presampling_strategy": "RandomPresamplingStrategy",
        },
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(0, 5000)),
            timestamps=list(range(100000, 105000)),
            labels=[1, 0] * 2500,
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(5000, 10000)),
            timestamps=list(range(200000, 205000)),
            labels=[0, 1] * 2500,
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 1, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []

    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        worker1_response = worker1_responses[0]
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)

        assert len(worker_1_samples) == 1000, f"Received {len(worker_1_samples)} samples instead of 1000."
        assert len(worker_2_samples) == 1000, f"Received {len(worker_2_samples)} samples instead of 1000."

        worker_1_weights = list(worker1_response.training_samples_weights)
        worker_2_weights = list(worker2_response.training_samples_weights)
        assert len(worker_1_samples) == len(worker_1_weights)
        assert len(worker_2_samples) == len(worker_2_weights)

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert len(total_samples) == len(set(total_samples)), "Received duplicated samples"
    assert set(total_samples) <= set(
        range(10000)
    ), f"Got samples with out of range keys: {set(total_samples) - set(range(10000))}"
    assert len(total_samples) == 2000, f"expected 2000 samples, got {len(total_samples)}"

    next_trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(10000, 15000)),
            timestamps=list(range(20000, 25000)),
            labels=list(range(20000, 25000)),
        )
    ).trigger_id

    assert next_trigger_id > trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=next_trigger_id)
    ).num_partitions

    assert number_of_partitions == 1, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []

    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=0, partition_id=partition
                )
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=1, partition_id=partition
                )
            )
        )

        worker1_response = worker1_responses[0]
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)

        if not reset_after_trigger:
            # we should have 0.2*15000 = 3000 points . So 1500 per worker
            assert len(worker_1_samples) == 1500, f"Received {len(worker_1_samples)}," f"instead of 1500."
            assert len(worker_2_samples) == 1500, f"Received {len(worker_2_samples)} instead of 1500."
        else:
            # we should have 0.2*5000 = 1000 points, so 500 per worker
            assert len(worker_1_samples) == 500, f"Received {len(worker_1_samples)}" f"instead of 500."
            assert len(worker_2_samples) == 500, f"Received {len(worker_2_samples)} instead of 500."

        worker_1_weights = list(worker1_response.training_samples_weights)
        worker_2_weights = list(worker2_response.training_samples_weights)
        assert len(worker_1_samples) == len(worker_1_weights)
        assert len(worker_2_samples) == len(worker_2_weights)

        total_samples.extend(worker_1_samples + worker_2_samples)

        if not reset_after_trigger:
            # ids must belong to [0,15000)
            assert set(total_samples) <= set(range(15000)), (
                f"Got {total_samples} but some samples do not belong to [0,15000). "
                f"Extra samples: {set(total_samples) - set(range(15000))}"
            )
            assert len(total_samples) == 3000, f"Expected 3000 samples, got {len(total_samples)}"

        else:
            # ids belong only to the last trigger [10000, 15000)
            assert set(total_samples) <= set(range(10000, 15000)), (
                f"Got {total_samples} but some samples do not belong to [10000,15000). "
                f"Extra samples: {set(total_samples) - set(range(10000, 15000))}"
            )
            assert len(total_samples) == 1000, f"Expected 1000 samples, got {len(total_samples)}"


def test_empty_triggers() -> None:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test without reset, i.e., after an empty trigger we get the same data

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 2,
        "config": {"limit": -1, "reset_after_trigger": False},
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[0, 1, 2],
            timestamps=[1, 2, 3],
            labels=[1, 0, 1],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[],
            timestamps=[],
            labels=[],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 2, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) <= 1
        assert len(worker_2_samples) <= 1

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(3)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 3

    next_trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[],
            timestamps=[],
            labels=[],
        )
    ).trigger_id

    assert next_trigger_id > trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 2, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) <= 1
        assert len(worker_2_samples) <= 1

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(3)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 3

    next_trigger_id2 = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[3, 4, 5],
            timestamps=[10, 11, 12],
            labels=[0, 0, 1],
        )
    ).trigger_id

    assert next_trigger_id2 > next_trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=next_trigger_id2)
    ).num_partitions

    assert number_of_partitions == 3, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id2, worker_id=0, partition_id=partition
                )
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(
                    pipeline_id=pipeline_id, trigger_id=next_trigger_id2, worker_id=1, partition_id=partition
                )
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) == 1
        assert len(worker_2_samples) == 1

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(6)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 6


def test_many_samples_evenly_distributed():
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test without reset, i.e., after an empty trigger we get the same data

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 5000,
        "config": {"limit": -1, "reset_after_trigger": False},
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(10000)),
            timestamps=list(range(10000)),
            labels=[0 for _ in range(10000)],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[],
            timestamps=[],
            labels=[],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 2, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        assert len(worker_1_samples) == 2500
        assert len(worker_2_samples) == 2500

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(10000)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 10000


def test_many_samples_unevenly_distributed():
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test without reset, i.e., after an empty trigger we get the same data

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 4999,
        "config": {"limit": -1, "reset_after_trigger": False},
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(10001)),
            timestamps=list(range(10001)),
            labels=[0 for _ in range(10001)],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[],
            timestamps=[],
            labels=[],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    assert number_of_partitions == 3, f"Invalid number of partitions: {number_of_partitions}"
    total_samples = []
    for partition in range(number_of_partitions):
        worker1_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition)
            )
        )

        worker2_responses: list[SamplesResponse] = list(
            selector.get_sample_keys_and_weights(
                GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition)
            )
        )

        assert len(worker1_responses) == 1
        worker1_response = worker1_responses[0]
        assert len(worker2_responses) == 1
        worker2_response = worker2_responses[0]

        worker_1_samples = list(worker1_response.training_samples_subset)
        worker_2_samples = list(worker2_response.training_samples_subset)
        if partition < number_of_partitions - 1:
            assert len(worker_1_samples) + len(worker_2_samples) == 4999
        else:
            assert len(worker_1_samples) + len(worker_2_samples) == 3

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert set(total_samples) == set(
        range(10001)
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(total_samples) == 10001


if __name__ == "__main__":
    test_newdata()
    test_empty_triggers()
    test_many_samples_evenly_distributed()
    test_many_samples_unevenly_distributed()
    test_abstract_downsampler(reset_after_trigger=False)
    test_abstract_downsampler(reset_after_trigger=True)
