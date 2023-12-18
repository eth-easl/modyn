import multiprocessing as mp
import queue
from typing import Optional

# less than client poll interval
QUEUE_GET_TIMEOUT = 0.1


class PipelineInfo:
    def __init__(
        self,
        process_handler: mp.Process,
        exception_queue: mp.Queue,
        pipeline_status_queue: mp.Queue,
        training_status_queue: mp.Queue,
        eval_status_queue: mp.Queue,
    ):
        self.process_handler = process_handler
        self.exception_queue = exception_queue

        self.pipeline_status_queue = pipeline_status_queue
        self.training_status_queue = training_status_queue
        self.eval_status_queue = eval_status_queue

    def get_msg_from_queue(self, queue_name: str, timeout: float = QUEUE_GET_TIMEOUT) -> Optional[dict]:
        try:
            # blocks for timeout seconds
            msg_queue = getattr(self, queue_name)
            return msg_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def check_for_exception(self, timeout: float = QUEUE_GET_TIMEOUT) -> Optional[str]:
        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = self.exception_queue.get(timeout=timeout)
            return exception
        except queue.Empty:
            return None
