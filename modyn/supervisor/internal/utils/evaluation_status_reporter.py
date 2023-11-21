import multiprocessing as mp


class EvaluationStatusReporter:
    def __init__(self, training_status_queue: mp.Queue, evaluation_id: int, dataset_id: str, dataset_size: int) -> None:
        self.training_status_queue = training_status_queue
        self.evaluation_id = evaluation_id
        self.dataset_id = dataset_id
        self.dataset_size = dataset_size
        self.last_samples_seen = 0

    def create_counter(self, training_id: int) -> None:
        desc = f"[Training {training_id}] Evaluation {self.evaluation_id} on dataset {self.dataset_id}"
        self.training_status_queue.put({"stage": "evaluation",
                                        "action": "create_counter", 
                                        "evaluation_id": self.evaluation_id,
                                        "training_id": training_id,
                                        "dataset_id": self.dataset_id,
                                        "dataset_size": self.dataset_size,
                                        "desc": desc})
        # self.counter = progress_mgr.counter(total=self.dataset_size, desc=desc, unit="samples", color="blue")

    def progress_counter(self, training_id: int, total_samples_seen: int) -> None:
        new_samples = total_samples_seen - self.last_samples_seen
        self.training_status_queue.put({"stage": "evaluation",
                                        "action": "progress_counter",
                                        "evaluation_id": self.evaluation_id,
                                        "training_id": training_id,
                                        "dataset_id": self.dataset_id,
                                        "new_samples": new_samples})
        # self.counter.update(new_samples)

        self.last_samples_seen = total_samples_seen

        if self.last_samples_seen == self.dataset_size:
            self.end_counter(training_id, False)

    def end_counter(self, training_id: int, error: bool) -> None:
        self.training_status_queue.put({"stage": "evaluation",
                                        "action": "end_counter", 
                                        "evaluation_id": self.evaluation_id,
                                        "training_id": training_id,
                                        "dataset_id": self.dataset_id,
                                        "error": error})
        # if not error:
        #     self.counter.update(self.counter.total - self.counter.count)
        # self.counter.clear(flush=True)
        # self.counter.close(clear=True)
