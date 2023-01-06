import pytest
from unittest.mock import patch

from modyn.trainer_server.utils.model_utils import get_model

class DummyModule:
    def __init__(self) -> None:
        self.model = lambda x: 10

def test_get_model_not_registered():
    with pytest.raises(ValueError):
        get_model("model", {})

@patch('modyn.trainer_server.utils.model_utils.dynamic_module_import')
def test_get_model(dynamic_module_patch):
    dynamic_module_patch.return_value = DummyModule()
    assert get_model("model", {}) == 10