import io
import pathlib
from typing import BinaryIO

import torch
from bitstring import BitArray

from modyn.model_storage.internal.storage_strategies.abstract_difference_operator import AbstractDifferenceOperator
from modyn.model_storage.internal.storage_strategies.difference_operators import (
    SubDifferenceOperator,
    XorDifferenceOperator,
)
from modyn.model_storage.internal.storage_strategies.incremental_model_strategies import (
    AbstractIncrementalModelStrategy,
)
from modyn.utils import get_tensor_byte_size

available_difference_operators: dict[str, type[AbstractDifferenceOperator]] = {
    "xor": XorDifferenceOperator,
    "sub": SubDifferenceOperator,
}


class WeightsDifference(AbstractIncrementalModelStrategy):
    """This incremental model strategy stores the delta between two successive
    model states as difference of their weight tensors.

    It currently supports two difference operators: xor and sub.
    """

    def __init__(self, zipping_dir: pathlib.Path, zip_activated: bool, zip_algorithm_name: str, config: dict):
        super().__init__(zipping_dir, zip_activated, zip_algorithm_name)

        self._validate_config(config)

    def _validate_config(self, config: dict) -> None:
        self.difference_operator: type[AbstractDifferenceOperator] = SubDifferenceOperator
        if "operator" in config:
            difference_operator_name = config["operator"]
            if difference_operator_name not in available_difference_operators:
                raise ValueError(f"Operator should be one of {available_difference_operators}.")
            self.difference_operator = available_difference_operators[difference_operator_name]
        self.split_exponent = config["split_exponent"] if "split_exponent" in config else False
        self.rle = config["rle"] if "rle" in config else False

    def _store_model(self, model_state: dict, prev_model_state: dict, file_path: pathlib.Path) -> None:
        bytestream = io.BytesIO()
        exponent_bytestream = io.BytesIO() if self.split_exponent else None

        for tensor_model, tensor_prev_model in zip(model_state.values(), prev_model_state.values()):
            difference = self.difference_operator.calculate_difference(tensor_model, tensor_prev_model)

            if exponent_bytestream is not None and tensor_model.dtype == torch.float32:
                for i in range(0, len(difference), 4):
                    reordered_diff = self.reorder_buffer(difference[i : i + 4])
                    bytestream.write(reordered_diff[0:3])
                    exponent_bytestream.write(reordered_diff[3:4])
            else:
                bytestream.write(difference)

        with open(file_path, "wb") as file:
            if exponent_bytestream is not None:
                exponents = exponent_bytestream.getvalue()
                if self.rle:
                    exponents = self.encode_bytes(exponents)
                file.write(len(exponents).to_bytes(8, byteorder="big"))
                file.write(exponents)
            file.write(bytestream.getbuffer().tobytes())

    def _load_model(self, prev_model_state: dict, file_path: pathlib.Path) -> dict:
        with open(file_path, "rb") as file:
            if not self.split_exponent:
                for layer_name, tensor in prev_model_state.items():
                    num_bytes = get_tensor_byte_size(tensor)
                    prev_model_state[layer_name] = self.difference_operator.restore(tensor, file.read(num_bytes))
                return prev_model_state
            return self._load_model_split_exponent(prev_model_state, file)

    def _load_model_split_exponent(self, prev_model_state: dict, file: BinaryIO) -> dict:
        exponent_bytes_amount = int.from_bytes(file.read(8), byteorder="big")

        with io.BytesIO() as exponent_bytes:
            exponent_bytes.write(
                self.decode_bytes(file.read(exponent_bytes_amount)) if self.rle else file.read(exponent_bytes_amount)
            )
            exponent_bytes.seek(0)

            for layer_name, tensor in prev_model_state.items():
                num_bytes = get_tensor_byte_size(tensor)

                if tensor.dtype == torch.float32:
                    buffer = bytearray(num_bytes)
                    for i in range(0, num_bytes, 4):
                        buffer[i : i + 3] = file.read(3)
                        buffer[i + 3 : i + 4] = exponent_bytes.read(1)

                    prev_model_state[layer_name] = self.difference_operator.restore(tensor, self.reorder_buffer(buffer))
                else:
                    prev_model_state[layer_name] = self.difference_operator.restore(tensor, file.read(num_bytes))
        return prev_model_state

    @staticmethod
    def reorder_buffer(buffer: bytes | bytearray) -> bytes:
        bit_array = BitArray(buffer)
        array_size = len(bit_array)

        for i in range(0, array_size, 32):
            # exchange sign bit with last exponent bit
            sign_bit = bit_array[i + 24]
            bit_array[i + 24] = bit_array[i + 16]
            bit_array[i + 16] = sign_bit

        return bit_array.bytes

    @staticmethod
    def encode_bytes(buffer: bytes) -> bytes:
        """Perform byte-wise run-length encoding.

        Args:
            buffer: the bytes to be encoded.

        Returns:
            bytes: the encoded bytes.
        """
        if len(buffer) == 0:
            return buffer
        bytestream = io.BytesIO()

        curr = buffer[0]
        count = 0

        for byte in buffer:
            if byte == curr and count < 255:
                count += 1
            else:
                bytestream.write(count.to_bytes(1, byteorder="big"))
                bytestream.write(curr.to_bytes(1, byteorder="big"))
                curr = byte
                count = 1
        bytestream.write(count.to_bytes(1, byteorder="big"))
        bytestream.write(curr.to_bytes(1, byteorder="big"))

        return bytestream.getvalue()

    @staticmethod
    def decode_bytes(buffer: bytes) -> bytes:
        """Decode run-length encoded bytes.

        Args:
            buffer: the encoded bytes.

        Returns:
            bytes: the decoded bytes.
        """
        assert len(buffer) % 2 == 0, "should be of even length"
        bytestream = io.BytesIO()

        for i in range(0, len(buffer), 2):
            count = int.from_bytes(buffer[i : i + 1], byteorder="big")

            bytestream.write(count * buffer[i + 1 : i + 2])
        return bytestream.getvalue()
