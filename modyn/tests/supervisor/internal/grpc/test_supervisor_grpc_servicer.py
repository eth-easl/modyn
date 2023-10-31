# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name

import tempfile
from typing import Iterable
from unittest.mock import MagicMock, patch

from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import PipelineResponse, StartPipelineRequest  # noqa: E402, E501, E611;
from modyn.supervisor.internal.grpc.supervisor_grpc_servicer import SupervisorGRPCServicer
from modyn.supervisor.supervisor import Supervisor


def get_minimal_modyn_config():
    return {"selector": {"keys_in_selector_cache": 1000, "trigger_sample_directory": "/does/not/exist"}}


def noop_init_metadata_db(self) -> None:
    pass


@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        sup = Supervisor(config)
        servicer = SupervisorGRPCServicer(sup, 8096)
        assert servicer.selector_manager == sup


@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
@patch.object(Supervisor, "start_pipeline")
def test_start_pipeline(test_start_pipeline: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        sup = Supervisor(config)
        servicer = SupervisorGRPCServicer(sup)
        # TODO(#317): prepare params
        request = StartPipelineRequest(pipeline_config={}, eval_directory=".", start_replay_at=0, stop_replay_at=1, maximum_triggers=2)
        test_start_pipeline.return_value = 1

        responses: Iterable[PipelineResponse] = list(servicer.start_pipeline(request, None))
        assert len(responses) == 1
        response = responses[0]
        assert response.pipeline_id == 1

        test_start_pipeline.assert_called_once_with({}, ".", 0, 1, 2)
