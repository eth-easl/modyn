from typing import Any

from modyn.utils import dynamic_module_import


def get_model(
    model_id: str,
    model_conf_dict: dict[str, Any],
    device: int,
) -> Any:

    """
    Gets handler to the model specified by the 'model_id'.
    The model should exist in the path "modyn/models/model_id"

    Returns:
        the requested model

    """

    model_module = dynamic_module_import("modyn.models")
    if not hasattr(model_module, model_id):
        raise ValueError(f"Model {model_id} not available!")

    model_handler = getattr(model_module, model_id)

    model = model_handler(model_conf_dict, device)

    return model
