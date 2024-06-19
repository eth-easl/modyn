import time
from unittest.mock import patch

from modyn.common.benchmark import Stopwatch
from modyn.trainer_server.internal.trainer.maybe_measure_gpu_ops import MaybeMeasureGPUOps


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.torch.cuda.synchronize")
def test_maybe_measure_gpu_ops(cuda_synchronize_mock):
    stopwatch = Stopwatch()
    with (
        patch.object(Stopwatch, "stop", wraps=stopwatch.stop) as stopwatch_stop_mock,
        patch.object(Stopwatch, "start", wraps=stopwatch.start) as stopwatch_start_mock,
    ):

        with MaybeMeasureGPUOps(True, "measure", "cpu", stopwatch, resume=True):
            time.sleep(1)

        stopwatch_start_mock.assert_called_once_with(name="measure", resume=True)
        stopwatch_stop_mock.assert_called_once_with(name="measure")
        assert 1000 <= stopwatch.measurements["measure"] <= 1100

        stopwatch_start_mock.reset_mock()
        stopwatch_stop_mock.reset_mock()
        with MaybeMeasureGPUOps(False, "measure2", "cpu", stopwatch, resume=False):
            time.sleep(1)

        stopwatch_start_mock.assert_not_called()
        stopwatch_stop_mock.assert_not_called()
        assert "measure2" not in stopwatch.measurements
