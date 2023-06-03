import multiprocessing as mp


class EvaluationProcessInfo:
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
