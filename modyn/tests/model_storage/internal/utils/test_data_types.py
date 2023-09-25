import io

import torch
from modyn.model_storage.internal.utils import read_tensor_from_bytes


def test_read_tensor_from_bytes():
    buf = io.BytesIO()
    buf.write(b"\x01\x00\x00\x00")
    buf.write(b"\x02\x00\x00\x00")
    buf.write(b"\x03\x00\x00\x00")
    buf.write(b"\x04\x00\x00\x00")
    buf.seek(0)
    res = read_tensor_from_bytes(torch.ones((2, 2), dtype=torch.int32), buf.getvalue())

    assert res[0, 0] == 1 and res[0, 1] == 2 and res[1, 0] == 3 and res[1, 1] == 4
