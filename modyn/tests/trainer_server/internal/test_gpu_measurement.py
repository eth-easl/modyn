import time
from unittest.mock import patch

from modyn.common.benchmark import Stopwatch
from modyn.trainer_server.internal.trainer.gpu_measurement import GPUMeasurement


@patch("modyn.trainer_server.internal.trainer.pytorch_trainer.torch.cuda.synchronize")
def test_gpu_measurement(cuda_synchronize_mock):
    stopwatch = Stopwatch()
    with (
        patch.object(Stopwatch, "stop", wraps=stopwatch.stop) as stopwatch_stop_mock,
        patch.object(Stopwatch, "start", wraps=stopwatch.start) as stopwatch_start_mock,
    ):
        with GPUMeasurement(True, "measure", "cpu", stopwatch, resume=True):
            time.sleep(1)

        stopwatch_start_mock.assert_called_once_with(name="measure", resume=True)
        stopwatch_stop_mock.assert_called_once_with(name="measure")
        assert cuda_synchronize_mock.call_count == 2
        assert 1000 <= stopwatch.measurements["measure"] <= 1100

        stopwatch_start_mock.reset_mock()
        stopwatch_stop_mock.reset_mock()
        cuda_synchronize_mock.reset_mock()
        with GPUMeasurement(False, "measure2", "cpu", stopwatch, overwrite=False):
            pass

        stopwatch_start_mock.assert_called_once_with(name="measure2", overwrite=False)
        stopwatch_stop_mock.assert_called_once_with(name="measure2")
        assert cuda_synchronize_mock.call_count == 0
        # we still want to take the (inaccurate) measurement
        assert "measure2" in stopwatch.measurements
