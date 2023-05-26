from typing import Optional

from modyn.utils import current_time_millis


class Stopwatch:
    """Stopwatch to be used in benchmarking.

    Using a stopwatch, you can do several named measurements using a start/stop
    interface and export all measurements as a dictionary.
    """

    def __init__(self):
        self.measurements: dict[str, int] = {}  # Unit is milliseconds
        self._running_measurements: dict[str, int] = {}
        self._last_started_measurement: Optional[str] = None

    def start(self, name: str) -> None:
        assert name not in self.measurements, "Measurement already done"
        assert name not in self._running_measurements, "Measurement already running"

        self._running_measurements[name] = current_time_millis()
        self._last_started_measurement = name

    def stop(self, name: Optional[str] = None) -> int:
        time = current_time_millis()

        if name is None:
            assert self._last_started_measurement is not None, "Cannot stop before starting a measurement"
            name = self._last_started_measurement

        assert name in self._running_measurements, "Measurement not running"
        assert name not in self.measurements, "Measurement already done"

        self.measurements[name] = time - self._running_measurements[name]

        if name == self._last_started_measurement:
            self._last_started_measurement = None

        return self.measurements[name]
