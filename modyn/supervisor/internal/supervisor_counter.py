from enum import Enum
from typing import Optional

import enlighten

CurrentEvent = Enum("CurrentEvent", ["IDLE", "TRAINING", "DOWNSAMPLING"])


class SupervisorCounter:
    def __init__(self, progress_mgr: enlighten.Manager, training_id: int) -> None:
        self.last_samples = 0
        self.current_epoch = 0
        self.epoch_samples = 0
        self.training_id = training_id
        self.current_event: CurrentEvent = CurrentEvent.IDLE
        self.progress_mgr: enlighten.Manager = progress_mgr
        self.sample_pbar: Optional[enlighten.counter] = None

    def start_training(self, samples_per_epoch: int) -> None:
        assert self.current_event == CurrentEvent.IDLE

        self.current_event = CurrentEvent.TRAINING
        self.sample_pbar = self._get_new_counter(samples_per_epoch)

        self.epoch_samples = samples_per_epoch

    def end_training(self) -> None:
        assert self.current_event != CurrentEvent.IDLE
        assert self.sample_pbar is not None

        self.sample_pbar.update(self.sample_pbar.total - self.sample_pbar.count)
        self.sample_pbar.clear(flush=True)
        self.sample_pbar.close(clear=True)

    def progress_counter(self, samples_seen: int) -> None:
        print(samples_seen)
        assert self.current_event != CurrentEvent.IDLE
        assert self.sample_pbar is not None
        assert samples_seen >= self.last_samples

        new_samples = samples_seen - self.last_samples

        if new_samples == 0:
            return

        next_epoch_boundary = self.epoch_samples * (self.current_epoch + 1)

        if samples_seen < next_epoch_boundary:
            self._progress_epoch(new_samples)
        elif samples_seen == next_epoch_boundary:
            self._end_epoch()
        else:
            this_epoch_samples = self.epoch_samples - self.sample_pbar.count
            self._progress_epoch(this_epoch_samples)
            self._end_epoch()
            print(self.current_epoch)

            remaining_samples = new_samples - self.epoch_samples

            while True:
                if remaining_samples >= self.epoch_samples:
                    self._progress_epoch(self.epoch_samples)
                    self._end_epoch()
                    remaining_samples -= self.epoch_samples
                else:
                    self._progress_epoch(remaining_samples)
                    break

        self.last_samples = samples_seen

    def _progress_epoch(self, samples: int) -> None:
        assert self.current_event != CurrentEvent.IDLE
        assert self.sample_pbar is not None

        self.sample_pbar.update(samples)

    def _end_epoch(self) -> None:
        assert self.current_event != CurrentEvent.IDLE
        assert self.sample_pbar is not None

        self.sample_pbar.update(self.sample_pbar.total - self.sample_pbar.count)
        self.sample_pbar.clear(flush=True)
        self.sample_pbar.close(clear=True)
        self.current_epoch += 1
        self.sample_pbar = self._get_new_counter(self.epoch_samples)

    def _get_new_counter(self, total_samples: int) -> enlighten.counter:
        assert self.current_event != CurrentEvent.IDLE

        if self.current_event == CurrentEvent.TRAINING:
            color = "red"
            desc = f"[Training {self.training_id} Epoch {self.current_epoch}] Training on Samples"
        elif self.current_event == CurrentEvent.DOWNSAMPLING:
            color = "green"
            desc = f"[Training {self.training_id} Epoch {self.current_epoch}] Downsampling"

        return self.progress_mgr.counter(total=total_samples, desc=desc, unit="samples", color=color)
