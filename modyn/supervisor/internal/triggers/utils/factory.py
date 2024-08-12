from modyn.config.schema.base_model import ModynBaseModel
from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.utils import dynamic_module_import


def instantiate_trigger(trigger_id: str, trigger_config: ModynBaseModel) -> Trigger:
    trigger_module = dynamic_module_import("modyn.supervisor.internal.triggers")
    trigger: Trigger = getattr(trigger_module, trigger_id)(trigger_config)
    assert trigger is not None, "Error during trigger initialization"
    return trigger
