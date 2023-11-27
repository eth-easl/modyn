import multiprocessing as mp


class TrainingStatusReporter:
    def __init__(
        self,
        training_status_queue: mp.Queue,
        trigger_id: int,
        training_id: int,
        samples_per_epoch: int,
        status_bar_scale: float,
    ) -> None:
        self.training_status_queue = training_status_queue
        self.trigger_id = trigger_id
        self.training_id = training_id
        self.samples_per_epoch = samples_per_epoch
        self.status_bar_scale = status_bar_scale
        self.stage = "wait for training"

    def _template_msg(self, action: str) -> None:
        return {"stage": self.stage, "action": action, "id": self.training_id}

    def create_tracker(self) -> None:
        ret = self._template_msg("create_tracker")
        ret["training_create_tracker_params"] = {
            "total_samples": self.samples_per_epoch,
            "status_bar_scale": self.status_bar_scale,
        }
        self.training_status_queue.put(ret)

    def progress_counter(self, samples_seen_training: int, samples_seen_downsampling: int, is_training: bool) -> None:
        ret = self._template_msg("progress_counter")
        ret["training_progress_counter_params"] = {
            "samples_seen": samples_seen_training,
            "downsampling_samples_seen": samples_seen_downsampling,
            "is_training": is_training,
        }
        self.training_status_queue.put(ret)

    def close_counter(self) -> None:
        ret = self._template_msg("close_counter")
        self.training_status_queue.put(ret)
