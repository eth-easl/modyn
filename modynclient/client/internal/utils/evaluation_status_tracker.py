from enlighten import Counter, Manager


class EvaluationStatusTracker:
    def __init__(self, dataset_id: str, dataset_size: int) -> None:
        """
        Class used to track the evaluation-progress of a dataset for the supervisor.

        Args:
            dataset_id: id of the dataset
            dataset_size: total size of the dataset
        """
        self.dataset_id = dataset_id
        self.dataset_size = dataset_size
        self.counter: Counter = None
        self.last_samples_seen = 0

    def create_counter(self, progress_mgr: Manager, training_id: int, evaluation_id: int) -> None:
        desc = f"[Training {training_id}] Evaluation {evaluation_id} on dataset {self.dataset_id}"
        self.counter = progress_mgr.counter(total=self.dataset_size, desc=desc, unit="samples", color="blue")

    def progress_counter(self, total_samples_seen: int) -> None:
        assert self.counter

        new_samples = total_samples_seen - self.last_samples_seen
        self.counter.update(new_samples)

        self.last_samples_seen = total_samples_seen

        if self.last_samples_seen == self.dataset_size:
            self.end_counter(False)

    def end_counter(self, error: bool) -> None:
        assert self.counter

        if not error:
            self.counter.update(self.counter.total - self.counter.count)
        self.counter.clear(flush=True)
        self.counter.close(clear=True)
