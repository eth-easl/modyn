import multiprocessing as mp


class EvaluationStatusReporter:
    def __init__(self, training_status_queue: mp.Queue, evaluation_id: int, dataset_id: str, dataset_size: int) -> None:
        self.training_status_queue = training_status_queue
        self.evaluation_id = evaluation_id
        self.dataset_id = dataset_id
        self.dataset_size = dataset_size

    def create_tracker(self) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "create_tracker",
                "id": self.evaluation_id,
                "eval_create_tracker_params": {
                    "dataset_id": self.dataset_id,
                    "dataset_size": self.dataset_size,
                },
            }
        )

    def create_counter(self, training_id: int) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "create_counter",
                "id": self.evaluation_id,
                "eval_create_counter_params": {
                    "training_id": training_id,
                },
            }
        )

    def progress_counter(self, total_samples_seen: int) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "progress_counter",
                "id": self.evaluation_id,
                "eval_progress_counter_params": {
                    "total_samples_seen": total_samples_seen,
                },
            }
        )

    def end_counter(self, error: bool) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "end_counter",
                "id": self.evaluation_id,
                "eval_end_counter_params": {
                    "error": error,
                },
            }
        )
