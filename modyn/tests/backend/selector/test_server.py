
from unittest.mock import patch
from modyn.backend.selector.selector_server import serve, SelectorGRPCServer
from modyn.backend.selector.internal.grpc_handler import GRPCHandler
from modyn.backend.selector.selector_pb2 import RegisterTrainingRequest, GetSamplesRequest  # noqa: E402, E501

from collections import Counter

# pylint: skip-file

# We do not use the parameters in this empty mock constructor.


def noop_constructor_mock(self, config: dict):  # pylint: disable=unused-argument
    pass


@patch.object(GRPCHandler, '_init_metadata', return_value=None)
@patch.object(GRPCHandler, '_connection_established', return_value=True)
@patch.object(GRPCHandler, 'register_training', return_value=0)
@patch.object(GRPCHandler, 'get_info_for_training', return_value=(8, 1))
@patch.object(GRPCHandler, 'get_samples_by_metadata_query')
def test_prepare_training_set(test_get_samples_by_metadata_query, test_get_info_for_training,
                              test_register_training, test__connection_established, test__init_metadata):
    sample_cfg = {
        'selector': {
            'package': 'score_selector',
            'class': 'ScoreSelector',
            'port': '50056',
        },
        'metadata_database': {
            'port': "50054",
            'hostname': "backend",
        }
    }

    servicer = SelectorGRPCServer(sample_cfg)
    assert servicer.register_training(RegisterTrainingRequest(
        training_set_size=8, num_workers=1), None).training_id == 0

    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]
    all_scores = [1] * 8
    all_seens = [False] * 8

    test_get_samples_by_metadata_query.return_value = all_samples, all_scores, all_seens, all_classes, all_samples

    assert set(servicer.get_sample_keys(GetSamplesRequest(training_id=0, training_set_number=0, worker_id=0),
               None).training_samples_subset) == set(["a", "b", "c", "d", "e", "f", "g", "h"])

    # serve(sample_cfg)
