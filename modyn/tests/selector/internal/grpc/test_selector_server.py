# pylint: disable=unused-argument,redefined-outer-name
import tempfile
from unittest.mock import patch

from modyn.selector.internal.grpc.selector_server import SelectorGRPCServer
from modyn.selector.internal.selector_manager import SelectorManager


def noop_init_metadata_db(self):
    pass


def get_modyn_config():
    return {
        "selector": {
            "port": "1337",
            "keys_in_selector_cache": 1000,
            "sample_batch_size": 8096,
            "trigger_sample_directory": "/does/not/exist",
        }
    }


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        grpc_server = SelectorGRPCServer(config)
        assert grpc_server.modyn_config == config
        assert isinstance(grpc_server.selector_manager, SelectorManager)
