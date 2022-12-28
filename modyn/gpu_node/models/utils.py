import importlib
from typing import Any, Optional

import torch

from modyn.gpu_node.grpc.trainer_server_pb2 import RegisterTrainServerRequest
from modyn.models.base_model import BaseModel


def get_model(
    request: RegisterTrainServerRequest,
    optimizer_dict: dict[str, Any],
    model_conf_dict: dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    device: int
) -> BaseModel:

    """
    Gets handler to the model specified by the 'model_id'.
    The model should exist in the path "modyn/models/model_id",
    and be defined in a 'Model' class, which extends the 'BaseModel' class.

    Returns:
        BaseModel: the requested model

    """

    # model exists - has been validated by the supervisor
    model_module = importlib.import_module("modyn.models." + request.model_id)

    model = model_module.Model(
        request.torch_optimizer,
        optimizer_dict,
        model_conf_dict,
        train_dataloader,
        val_dataloader,
        device,
        request.checkpoint_info.checkpoint_path,
        request.checkpoint_info.checkpoint_interval,
    )

    return model
