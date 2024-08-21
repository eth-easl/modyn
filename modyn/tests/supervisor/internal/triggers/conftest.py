from pathlib import Path

import pytest

from modyn.config.schema.pipeline.config import ModynPipelineConfig
from modyn.config.schema.system.config import ModynConfig
from modyn.supervisor.internal.triggers.trigger import TriggerContext


@pytest.fixture
def dummy_trigger_context(
    dummy_system_config: ModynPipelineConfig, dummy_pipeline_config: ModynConfig
) -> TriggerContext:
    return TriggerContext(42, dummy_pipeline_config, dummy_system_config, Path("/tmp"))
