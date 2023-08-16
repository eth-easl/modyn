import io
import pathlib

from modyn.model_storage.internal.storage_strategies.difference_operators import (
    SubDifferenceOperator,
    XorDifferenceOperator,
)
from modyn.model_storage.internal.storage_strategies.incremental_model_strategies import (
    AbstractIncrementalModelStrategy,
)

available_difference_operators = {"xor": XorDifferenceOperator, "sub": SubDifferenceOperator}


class WeightsDifference(AbstractIncrementalModelStrategy):
    """
    This incremental model strategy stores the delta between two successive model states as difference of their
    weight tensors. It currently supports two difference operators: xor and sub.
    """

    def __init__(self, zip_activated: bool, zip_algorithm_name: str, config: dict):
        self.difference_operator = SubDifferenceOperator
        self.split_exponent = False

        super().__init__(zip_activated, zip_algorithm_name, config)

    def _save_model(self, model_state: dict, prev_model_state: dict, file_path: pathlib.Path) -> None:
        bytestream = io.BytesIO()

        for tensor_model, tensor_prev_model in zip(model_state.values(), prev_model_state.values()):
            bytestream.write(self.difference_operator.calculate_difference(tensor_model, tensor_prev_model))

        with open(file_path, "wb") as file:
            file.write(bytestream.getbuffer().tobytes())

    def _load_model(self, prev_model_state: dict, file_path: pathlib.Path) -> None:
        with open(file_path, "rb") as file:
            for layer_name, tensor in prev_model_state.items():
                prev_model_state[layer_name] = self.difference_operator.restore(tensor, file)

    def validate_config(self, config: dict) -> None:
        if "operator" in config:
            difference_operator_name = config["operator"]
            if difference_operator_name not in available_difference_operators:
                raise ValueError(f"Operator should be one of {available_difference_operators}.")
            self.difference_operator = available_difference_operators[difference_operator_name]
        self.split_exponent = config["split_exponent"] if "split_exponent" in config else False
