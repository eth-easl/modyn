import multiprocessing as mp


class EvaluationStatusReporter:
    def __init__(self, eval_status_queue: mp.Queue, evaluation_id: int, dataset_id: str, dataset_size: int) -> None:
        self.eval_status_queue = eval_status_queue
        self.evaluation_id = evaluation_id
        self.dataset_id = dataset_id
        self.dataset_size = dataset_size
        self.stage = "wait for evaluation"

    def _template_msg(self, action: str) -> None:
        return {"stage": self.stage, "action": action, "id": self.evaluation_id}

    def create_tracker(self) -> None:
        ret = self._template_msg("create_tracker")
        ret["eval_create_tracker_params"] = {
            "dataset_id": self.dataset_id,
            "dataset_size": self.dataset_size,
        }
        self.eval_status_queue.put(ret)

    def create_counter(self, training_id: int) -> None:
        ret = self._template_msg("create_counter")
        ret["eval_create_counter_params"] = {
            "training_id": training_id,
        }
        self.eval_status_queue.put(ret)

    def progress_counter(self, total_samples_seen: int) -> None:
        ret = self._template_msg("progress_counter")
        ret["eval_progress_counter_params"] = {
            "total_samples_seen": total_samples_seen,
        }
        self.eval_status_queue.put(ret)

    def end_counter(self, error: bool, exception_msg: str = "") -> None:
        ret = self._template_msg("end_counter")
        params = {"error": error}
        if error:
            params["exception_msg"] = exception_msg
        ret["eval_end_counter_params"] = params
        self.eval_status_queue.put(ret)
