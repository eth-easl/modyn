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
                "evaluation_id": self.evaluation_id,
                "dataset_id": self.dataset_id,
                "dataset_size": self.dataset_size,
            }
        )

    def create_counter(self, training_id: int) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "create_counter",
                "evaluation_id": self.evaluation_id,
                "training_id": training_id,
                "dataset_id": self.dataset_id,
                "dataset_size": self.dataset_size,
            }
        )

    def progress_counter(self, training_id: int, total_samples_seen: int) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "progress_counter",
                "evaluation_id": self.evaluation_id,
                "training_id": training_id,
                "dataset_id": self.dataset_id,
                "total_samples_seen": total_samples_seen,
            }
        )

    def end_counter(self, training_id: int, error: bool) -> None:
        self.training_status_queue.put(
            {
                "stage": "wait for evaluation",
                "action": "end_counter",
                "evaluation_id": self.evaluation_id,
                "training_id": training_id,
                "dataset_id": self.dataset_id,
                "error": error,
            }
        )
