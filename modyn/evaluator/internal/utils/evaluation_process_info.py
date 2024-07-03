import multiprocessing as mp


class EvaluationProcessInfo:
    def __init__(
        self,
        process_handler: mp.Process,
        exception_queue: mp.Queue,
        metric_result_queue: mp.Queue,
    ):
        self.process_handler = process_handler
        self.exception_queue = exception_queue
        self.metric_result_queue = metric_result_queue
