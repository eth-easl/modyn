import grpc
from integrationtests.utils import get_minimal_pipeline_config, get_modyn_config, init_metadata_db, register_pipeline
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    GetAvailableLabelsRequest,
    GetNumberOfPartitionsRequest,
    GetSamplesRequest,
    SamplesResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.utils import grpc_connection_established

# TODO(54): Write more integration tests for different strategies.


init_metadata_db(get_modyn_config())


def connect_to_selector_servicer() -> grpc.Channel:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

    return selector_channel


def test_label_balanced_presampling_huge() -> None:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "CoresetStrategy",
        "maximum_keys_in_memory": 250,
        "config": {
            "limit": -1,
            "reset_after_trigger": True,
            "presampling_config": {"strategy": "LabelBalancedPresamplingStrategy", "ratio": 50},
        },
    }

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=list(range(10, 5010)),
            timestamps=list(range(100000, 105000)),
            labels=[0, 1] * 2500,
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    # 5000 samples (reset after trigger) , 2500 sampled, 10 partitions
    assert number_of_partitions == 10, f"Invalid number of partitions: {number_of_partitions}"
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

        assert len(worker_1_samples) == 125, f"Received {len(worker_1_samples)} samples instead of 125."
        assert len(worker_2_samples) == 125, f"Received {len(worker_2_samples)} samples instead of 125."

        worker_1_weights = list(worker1_response.training_samples_weights)
        worker_2_weights = list(worker2_response.training_samples_weights)
        assert len(worker_1_samples) == len(worker_1_weights)
        assert len(worker_2_samples) == len(worker_2_weights)

        total_samples.extend(worker_1_samples + worker_2_samples)

    assert len(total_samples) == len(set(total_samples)), "Received duplicated samples"
    assert set(total_samples) <= set(
        range(10000)
    ), f"Got samples with out of range keys: {set(total_samples) - set(range(10000))}"
    assert len(total_samples) == 2500, f"expected 2000 samples, got {len(total_samples)}"

    # count of samples with label 1
    assert sum(1 if el % 2 == 0 else 0 for el in total_samples) == 1250
    # count of samples with label 0
    assert sum(1 if el % 2 == 1 else 0 for el in total_samples) == 1250


def test_label_balanced_force_same_size():
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "CoresetStrategy",
        "maximum_keys_in_memory": 100,
        "config": {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_config": {
                "strategy": "LabelBalancedPresamplingStrategy",
                "ratio": 90,
                "force_column_balancing": True,
            },
        },
    }

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

    # now we just have 2 classes with 4 samples each
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
            keys=[3, 4, 5, 6, 7],
            timestamps=[4, 5, 6, 7, 8],
            labels=[0, 1, 0, 1, 0],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    # We have 8 samples, 4 for each class. 90% presampling means 6 samples per class hence 1 partition
    assert number_of_partitions == 1, f"Invalid number of partitions: {number_of_partitions}"

    total_samples = _get_samples(pipeline_id, selector, trigger_id, 0, [3, 3])

    assert len(total_samples) == 6
    # 3 samples with label 0
    assert len(set(total_samples).intersection({1, 3, 5, 7})) == 3
    # 3 samples with label 1
    assert len(set(total_samples).intersection({0, 2, 4, 6})) == 3

    # now let's add a third, smaller class with just two samples
    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[100, 101],
            timestamps=[9, 10],
            labels=[2, 2],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    # We have 10 samples, [4,4,2] for each class. Since we want strong balance, each class should have 2 samples
    assert number_of_partitions == 1, f"Invalid number of partitions: {number_of_partitions}"

    total_samples = _get_samples(pipeline_id, selector, trigger_id, 0, [3, 3])

    assert len(total_samples) == 6
    # 2 samples with label 0
    assert len(set(total_samples).intersection({1, 3, 5, 7})) == 2
    # 2 samples with label 1
    assert len(set(total_samples).intersection({0, 2, 4, 6})) == 2
    # 2 samples with label 2
    assert len(set(total_samples).intersection({100, 101})) == 2


def test_label_balanced_force_all_samples():
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "CoresetStrategy",
        "maximum_keys_in_memory": 100,
        "config": {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_config": {
                "strategy": "LabelBalancedPresamplingStrategy",
                "ratio": 90,
                "force_required_target_size": True,
            },
        },
    }

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

    # same classes as before
    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[0, 1, 2],
            timestamps=[1, 2, 3],
            labels=[1, 0, 1],
        )
    )

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[3, 4, 5, 6, 7],
            timestamps=[4, 5, 6, 7, 8],
            labels=[0, 1, 0, 1, 0],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[100, 101],
            timestamps=[9, 10],
            labels=[2, 2],
        )
    ).trigger_id

    number_of_partitions = selector.get_number_of_partitions(
        GetNumberOfPartitionsRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
    ).num_partitions

    # We have 10 samples, [4,4,2] for each class. Target is 9 (90%)
    # We want to get exactly 9 samples
    assert number_of_partitions == 1, f"Invalid number of partitions: {number_of_partitions}"

    total_samples = _get_samples(pipeline_id, selector, trigger_id, 0, [5, 4])

    assert len(total_samples) == 9

    # 3 or 4 samples of class 0 (there is an extra sample to achieve exactly 9)
    assert 3 <= len(set(total_samples).intersection({1, 3, 5, 7})) <= 4
    # 3 or 4 samples of class 1 (there is an extra sample to achieve exactly 9)
    assert 3 <= len(set(total_samples).intersection({0, 2, 4, 6})) <= 4
    # 1 or 2 samples with label 2 (there is an extra sample to achieve exactly 9)
    assert 1 <= len(set(total_samples).intersection({100, 101})) <= 2


def _get_samples(pipeline_id, selector, trigger_id, partition_id, expected_len):
    worker1_responses: list[SamplesResponse] = list(
        selector.get_sample_keys_and_weights(
            GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0, partition_id=partition_id)
        )
    )
    worker2_responses: list[SamplesResponse] = list(
        selector.get_sample_keys_and_weights(
            GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1, partition_id=partition_id)
        )
    )
    worker1_response = worker1_responses[0]
    worker2_response = worker2_responses[0]
    worker_1_samples = list(worker1_response.training_samples_subset)
    worker_2_samples = list(worker2_response.training_samples_subset)
    assert (
        len(worker_1_samples) == expected_len[0]
    ), f"Received {len(worker_1_samples)} samples instead of {expected_len[0]}."
    assert (
        len(worker_2_samples) == expected_len[1]
    ), f"Received {len(worker_1_samples)} samples instead of {expected_len[1]}."
    total_samples = worker_1_samples + worker_2_samples
    assert len(total_samples) == sum(expected_len), f"Expected {sum(expected_len)} samples, got {len(total_samples)}"
    return total_samples


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

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

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
        "name": "CoresetStrategy",
        "maximum_keys_in_memory": 50000,
        "config": {
            "limit": -1,
            "reset_after_trigger": reset_after_trigger,
            "presampling_config": {"ratio": 20, "strategy": "Random"},
            "downsampling_config": {
                "ratio": 10,
                "strategy": "Loss",
                "sample_then_batch": False,
            },
        },
    }

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

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

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

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

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

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

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

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


def test_get_available_labels(reset_after_trigger: bool):
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 2,
        "config": {"limit": -1, "reset_after_trigger": reset_after_trigger},
    }

    pipeline_config = get_minimal_pipeline_config(2, strategy_config)
    pipeline_id = register_pipeline(pipeline_config, get_modyn_config())

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[0, 1, 2],
            timestamps=[1, 2, 3],
            labels=[1, 0, 1],
        )
    )
    available_labels = selector.get_available_labels(
        GetAvailableLabelsRequest(pipeline_id=pipeline_id)
    ).available_labels

    # here we expect to have 0 labels since it's before the first trigger
    assert len(available_labels) == 0

    selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[3],
            timestamps=[1],
            labels=[189],
        )
    )

    available_labels = selector.get_available_labels(
        GetAvailableLabelsRequest(pipeline_id=pipeline_id)
    ).available_labels

    # we want all the labels belonging to the first trigger
    assert len(available_labels) == 3
    assert sorted(available_labels) == [0, 1, 189]

    selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[4, 5, 6],
            timestamps=[4, 5, 6],
            labels=[10, 7, 45],
        )
    )

    # this label (99) should not appear in the available labels since it belongs to a future trigger.
    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[99],
            timestamps=[7],
            labels=[99],
        )
    )

    available_labels = selector.get_available_labels(
        GetAvailableLabelsRequest(pipeline_id=pipeline_id)
    ).available_labels

    if reset_after_trigger:
        # only the last trigger must be considered but not point99
        assert len(available_labels) == 3
        assert sorted(available_labels) == [7, 10, 45]
    else:
        # every past point must be considered. Only point99 is excluded.
        assert len(available_labels) == 6
        assert sorted(available_labels) == [0, 1, 7, 10, 45, 189]


if __name__ == "__main__":
    test_newdata()
    test_label_balanced_presampling_huge()
    test_label_balanced_force_same_size()
    test_label_balanced_force_all_samples()
    test_empty_triggers()
    test_many_samples_evenly_distributed()
    test_many_samples_unevenly_distributed()
    test_abstract_downsampler(reset_after_trigger=False)
    test_abstract_downsampler(reset_after_trigger=True)
    test_get_available_labels(reset_after_trigger=False)
    test_get_available_labels(reset_after_trigger=True)
