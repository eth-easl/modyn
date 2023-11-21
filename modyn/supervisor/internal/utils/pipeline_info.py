import multiprocessing as mp
import queue
from typing import Optional


class PipelineInfo:
    def __init__(
        self,
        process_handler: mp.Process,
        exception_queue: mp.Queue,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue,
    ):
        self.process_handler = process_handler
        self.exception_queue = exception_queue

        self.status_query_queue = status_query_queue
        self.status_response_queue = status_response_queue
    
    def get_status_detail(self, timeout: float = 10) -> dict:
        try:
            # blocks for timeout seconds
            return self.status_response_queue.get(timeout=timeout)
        except queue.Empty:
            return {}

    def check_for_exception(self) -> Optional[str]:
        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = self.exception_queue.get(timeout=0.1)
            return exception
        except queue.Empty:
            return None
