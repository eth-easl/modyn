import multiprocessing as mp


class TrainingStatusReporter:
    def __init__(
        self, training_status_queue: mp.Queue, trigger_id: int, training_id: int, samples_per_epoch: int, status_bar_scale: float
    ) -> None:
        self.training_status_queue = training_status_queue
        self.trigger_id = trigger_id
        self.training_id = training_id
        self.samples_per_epoch =  samples_per_epoch
        self.status_bar_scale = status_bar_scale
        
    def create_tracker(self) -> None:
        self.training_status_queue.put({
            "stage": "wait for training",
            "action": "create_tracker", 
            "trigger_id": self.trigger_id,
            "training_id": self.training_id, 
            "total_samples": self.samples_per_epoch,
            "status_bar_scale": self.status_bar_scale
        })

    def progress_counter(
        self, samples_seen_training: int, samples_seen_downsampling: int, is_training: bool
    ) -> None:
        self.training_status_queue.put({
            "stage":  "wait for training",
            "action": "progress_counter",
            "trigger_id": self.trigger_id,
            "training_id": self.training_id,
            "samples_seen": samples_seen_training,
            "downsampling_samples_seen": samples_seen_downsampling,
            "is_training": is_training
        })

    def close_counter(self) -> None:
        self.training_status_queue.put({
            "stage": "wait for training",
            "action": "close_counter",
            "trigger_id": self.trigger_id,
            "training_id": self.training_id
        })