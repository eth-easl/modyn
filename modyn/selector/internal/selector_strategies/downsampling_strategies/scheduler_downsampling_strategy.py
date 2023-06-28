from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies.utils import instantiate_downsampler


class SchedulerDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        if "first_strategy" not in downsampling_config:
            raise ValueError(
                "If you want to use a Downsampling Scheduler, please specify the first and the second strategy."
            )
        first_strategy = downsampling_config["first_strategy"]

        if "second_strategy" not in downsampling_config:
            raise ValueError(
                "If you want to use a Downsampling Scheduler, please specify the first and the second strategy."
            )
        second_strategy = downsampling_config["second_strategy"]

        if "threshold" not in downsampling_config:
            raise ValueError("Please specify after how many epochs you want to switch strategy.")
        self.threshold = downsampling_config["threshold"]

        if "ratio" in downsampling_config:
            raise ValueError(
                "SchedulerDownsamplingStrategy has no downsampling ratio. Specify the downsampling ratio"
                "inside first_strategy and second_strategy."
            )
        downsampling_config["ratio"] = 100  # just to deal with super

        if "sample_then_batch" in downsampling_config:
            raise ValueError("SchedulerDownsamplingStrategy has no downsampling mode (sample_then_batch parameter).")
        downsampling_config["sample_then_batch"] = True  # useless, just to avoid ValueErrors

        # instantiate the Scheduler
        super().__init__(downsampling_config, maximum_keys_in_memory)
        self.first_downsampler = instantiate_downsampler(
            {"downsampling_config": first_strategy}, maximum_keys_in_memory
        )
        self.second_downsampler = instantiate_downsampler(
            {"downsampling_config": second_strategy}, maximum_keys_in_memory
        )

        self.counter = 0

        # To count the elapsed epochs we can rely on how many calls were made to the functions below.
        # Calls can be generated from two components: Supervisor (get_training_status_bar_scale)
        # TrainerServer (everything else). SInce we have no guarantees on the order, and we want to avoid double
        # counting, these two variables are introduced.
        # Note that, from the trainer server, we just need to consider get_requires_remote_computation since the other
        # calls are made sequentially

        self._trainer_counter = 0
        self._supervisor_counter = 0

    def get_requires_remote_computation(self) -> bool:
        print(f"REQUIRES. Counter: {self.counter}, Sup: {self._supervisor_counter}, Tra: {self._trainer_counter}")

        if self._trainer_counter == self._supervisor_counter == self.counter:
            # increment the counters
            self._trainer_counter += 1
            self.counter += 1
        elif self._trainer_counter < self.counter and self.counter == self._supervisor_counter:
            self._trainer_counter += 1
            # no need to increment the counter since it was already incremented by the supervisor
        else:
            assert False, "Counting failed"

        if self.counter <= self.threshold:
            return self.first_downsampler.get_requires_remote_computation()
        return self.second_downsampler.get_requires_remote_computation()

    def get_training_status_bar_scale(self) -> int:
        print(f"SCALE. Counter: {self.counter}, Sup: {self._supervisor_counter}, Tra: {self._trainer_counter}")

        if self._trainer_counter == self._supervisor_counter == self.counter:
            # increment the counters
            self._supervisor_counter += 1
            self.counter += 1
        elif self._supervisor_counter < self.counter and self.counter == self._trainer_counter:
            self._supervisor_counter += 1
            # no need to increment the counter since it was already incremented by the trainer
        else:
            assert False, "Counting failed"

        if self.counter <= self.threshold:
            return self.first_downsampler.get_training_status_bar_scale()
        return self.second_downsampler.get_training_status_bar_scale()

    def get_downsampling_strategy(self) -> str:
        print(f"STRATEGY. Counter: {self.counter}, Sup: {self._supervisor_counter}, Tra: {self._trainer_counter}")

        if self.counter <= self.threshold:
            return self.first_downsampler.get_downsampling_strategy()
        return self.second_downsampler.get_downsampling_strategy()

    def get_downsampling_params(self) -> dict:
        print(f"PARAMS. Counter: {self.counter}, Sup: {self._supervisor_counter}, Tra: {self._trainer_counter}")

        if self.counter <= self.threshold:
            return self.first_downsampler.get_downsampling_params()
        return self.second_downsampler.get_downsampling_params()
