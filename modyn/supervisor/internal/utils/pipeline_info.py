import multiprocessing as mp
import queue
from enum import Enum
from typing import Optional


class PipelineExitCode(Enum):
    NULL = 1
    SUCCESS = 2
    ERROR = 3


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

        # TODO(#317): maybe use status enum to report fine-grained stages
        self.pipeline_exited = False
        self.pipeline_exit_code = PipelineExitCode.NULL
        self.exception_msg: Optional[str] = None

    def check_for_exception(self) -> Optional[str]:
        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = self.exception_queue.get(timeout=0.1)
            return exception
        except queue.Empty:
            return None

    # TODO(#317): regard ERROR as pipeline stopped and implement process GC
    # TODO(#317): also check status queue?
    def update_status(self) -> None:
        if not self.pipeline_exited:
            self.exception_msg = self.check_for_exception()
            # Mark pipeline as exited when exception occurs
            # But, process is not dead right after exception, it is alive until joined with the parent process
            if self.exception_msg is not None:
                self.pipeline_exited = True
                self.pipeline_exit_code = PipelineExitCode.ERROR
            # Mark pipeline as exited when process is joined w.o. exception
            elif not self.process_handler.is_alive():
                self.pipeline_exited = True
                self.pipeline_exit_code = PipelineExitCode.SUCCESS
