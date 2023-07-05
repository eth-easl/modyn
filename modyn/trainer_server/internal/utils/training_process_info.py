import multiprocessing as mp


class TrainingProcessInfo:
    def __init__(
        self,
        process_handler: mp.Process,
        exception_queue: mp.Queue,
        status_query_queue_training: mp.Queue,
        status_response_queue_training: mp.Queue,
        status_query_queue_downsampling: mp.Queue,
        status_response_queue_downsampling: mp.Queue,
    ):
        self.process_handler = process_handler
        self.exception_queue = exception_queue

        self.status_query_queue_training = status_query_queue_training
        self.status_response_queue_training = status_response_queue_training

        self.status_query_queue_downsampling = status_query_queue_downsampling
        self.status_response_queue_downsampling = status_response_queue_downsampling
        self.was_training = True
