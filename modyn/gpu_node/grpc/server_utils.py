from typing import Any
from modyn.gpu_node.grpc.trainer_server_pb2 import VarTypeParameter


def process_grpc_map(grpc_dict: dict[str, VarTypeParameter]) -> dict[str, Any]:

    """
    Converts grpc map to python dict with proper casting.

    Returns:
        dict[str, Any]: a dict with values of different types.
    """

    new_dict = {}
    for key, value in grpc_dict.items():
        if value.HasField("float_value"):
            new_value = value.float_value
        elif value.HasField("int_value"):
            new_value = value.int_value
        elif value.HasField("bool_value"):
            new_value = value.bool_value
        else:
            new_value = value.string_value
        new_dict[key] = new_value

    return new_dict
