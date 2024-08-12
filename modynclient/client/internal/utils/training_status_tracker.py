from enum import Enum

import enlighten

CurrentEvent = Enum("CurrentEvent", ["IDLE", "TRAINING", "DOWNSAMPLING"])


class TrainingStatusTracker:
    def __init__(
        self, progress_mgr: enlighten.Manager, training_id: int, samples_per_epoch: int, status_bar_scale: float
    ) -> None:
        """
        Class to handle the progress bar in the supervisor panel.
        Args:
            progress_mgr: progress manager
            training_id: training id (shown in the progress bar)
            samples_per_epoch: total number of samples returned by the selector
            status_bar_scale: downsampling performed on samples_per_epoch samples, training on
                            status_bar_scale * samples_per_epoch samples
        """
        self.last_samples_training = 0
        self.last_samples_downsampling = 0
        self.current_epoch = 0

        self.num_samples_downsampling = samples_per_epoch
        self.num_samples_epoch_training = int(status_bar_scale * samples_per_epoch / 100)

        self.training_id = training_id
        self.current_event: CurrentEvent = CurrentEvent.IDLE
        self.progress_mgr: enlighten.Manager = progress_mgr
        self.sample_pbar: enlighten.counter | None = None

    def progress_counter(self, samples_seen_training: int, samples_seen_downsampling: int, is_training: bool) -> None:
        if self.current_event == CurrentEvent.IDLE:
            if is_training:
                self._start_training_epoch()
            else:
                self._start_downsampling()

        if self.current_event == CurrentEvent.DOWNSAMPLING and is_training:
            self._end_downsampling()
            self._start_training_epoch()

        if self.current_event == CurrentEvent.TRAINING and not is_training:
            self._end_training_epoch()
            self._start_downsampling()

        if self.current_event == CurrentEvent.TRAINING:
            if samples_seen_training < self.last_samples_training:
                # outdated message
                return
            self._progress_counter_training(samples_seen_training)

        elif self.current_event == CurrentEvent.DOWNSAMPLING:
            self._progress_counter_downsampling(samples_seen_downsampling)

    def close_counter(self) -> None:
        """Close the last open progress bar, if any."""
        if self.current_event == CurrentEvent.IDLE:
            return

        assert self.sample_pbar is not None
        self.sample_pbar.update(self.sample_pbar.total - self.sample_pbar.count)
        self.sample_pbar.clear(flush=True)
        self.sample_pbar.close(clear=True)
        self.current_event = CurrentEvent.IDLE
        self.sample_pbar = None

    def _start_downsampling(self) -> None:
        assert self.current_event == CurrentEvent.IDLE
        assert self.sample_pbar is None

        self.last_samples_downsampling = 0
        self.current_event = CurrentEvent.DOWNSAMPLING
        self.sample_pbar = self._get_new_counter(self.num_samples_downsampling)

    def _end_downsampling(self) -> None:
        assert self.current_event == CurrentEvent.DOWNSAMPLING
        assert self.sample_pbar is not None

        self.sample_pbar.update(self.sample_pbar.total - self.sample_pbar.count)
        self.sample_pbar.clear(flush=True)
        self.sample_pbar.close(clear=True)
        self.current_event = CurrentEvent.IDLE
        self.sample_pbar = None

    def _progress_counter_training(self, samples_seen_training: int) -> None:
        assert self.current_event == CurrentEvent.TRAINING
        assert self.sample_pbar is not None

        # samples that must be added to the progress bar
        new_samples = samples_seen_training - self.last_samples_training

        if new_samples == 0:
            return

        # samples before reaching a new epoch
        next_epoch_boundary = self.num_samples_epoch_training * (self.current_epoch + 1)

        if samples_seen_training < next_epoch_boundary:
            # if we don't cross an epoch boundary, we can just increase the current progress bar
            self._progress_epoch(new_samples)
        elif samples_seen_training == next_epoch_boundary:
            # increase and close the progress bar
            self._progress_epoch(new_samples)
            self._end_training_epoch()
        else:
            # more than one epoch

            # finish the current epoch
            this_epoch_samples = self.num_samples_epoch_training - self.sample_pbar.count
            self._progress_epoch(this_epoch_samples)
            self._end_training_epoch()

            # start a new epoch
            self._start_training_epoch()
            remaining_samples = new_samples - this_epoch_samples

            while True:
                if remaining_samples > self.num_samples_epoch_training:
                    # more than one epoch, finish this one and start a new one
                    self._progress_epoch(self.num_samples_epoch_training)
                    self._end_training_epoch()
                    self._start_training_epoch()
                    remaining_samples -= self.num_samples_epoch_training
                elif remaining_samples == self.num_samples_epoch_training:
                    # just finish this one
                    self._progress_epoch(remaining_samples)
                    self._end_training_epoch()
                    break
                else:
                    # progress the current epoch
                    self._progress_epoch(remaining_samples)
                    break

        self.last_samples_training = samples_seen_training

    def _progress_counter_downsampling(self, samples_seen_downsampling: int) -> None:
        """
        Handling the downsampling progress bar. Since there's no concept of epoch in downsampling,
        the logic is way easier
        Args:
            samples_seen_downsampling: samples that must be added to the progress bar

        Returns:

        """
        assert self.current_event == CurrentEvent.DOWNSAMPLING
        assert self.sample_pbar is not None

        new_samples = samples_seen_downsampling - self.last_samples_downsampling

        self.sample_pbar.update(new_samples)

        self.last_samples_downsampling = samples_seen_downsampling

        if self.last_samples_downsampling == self.num_samples_downsampling:
            self._end_downsampling()

    def _start_training_epoch(self) -> None:
        assert self.current_event == CurrentEvent.IDLE
        assert self.sample_pbar is None

        self.current_event = CurrentEvent.TRAINING
        self.sample_pbar = self._get_new_counter(self.num_samples_epoch_training)
        assert self.current_event == CurrentEvent.TRAINING

    def _progress_epoch(self, samples: int) -> None:
        assert self.current_event == CurrentEvent.TRAINING
        assert self.sample_pbar is not None

        self.sample_pbar.update(samples)

    def _end_training_epoch(self) -> None:
        assert self.current_event == CurrentEvent.TRAINING
        assert self.sample_pbar is not None

        self.sample_pbar.update(self.sample_pbar.total - self.sample_pbar.count)
        self.sample_pbar.clear(flush=True)
        self.sample_pbar.close(clear=True)
        self.current_epoch += 1
        self.sample_pbar = None
        self.current_event = CurrentEvent.IDLE

    def _get_new_counter(self, total_samples: int) -> enlighten.counter:
        assert self.current_event != CurrentEvent.IDLE

        if self.current_event == CurrentEvent.TRAINING:
            color = "red"
            desc = f"[Training {self.training_id} Epoch {self.current_epoch}] Training on Samples"
        elif self.current_event == CurrentEvent.DOWNSAMPLING:
            color = "green"
            desc = f"[Training {self.training_id} Epoch {self.current_epoch}] Downsampling"

        return self.progress_mgr.counter(total=total_samples, desc=desc, unit="samples", color=color)
