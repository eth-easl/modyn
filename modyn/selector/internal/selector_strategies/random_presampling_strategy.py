from modyn.selector.internal.selector_strategies.abstract_presample_strategy import AbstractPresampleStrategy


class RandomPresamplingStrategy(AbstractPresampleStrategy):
    """
    Class to train on a sampled subset of the whole dataset
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)
        if "downsampled_batch_ratio" in config:
            raise ValueError(
                "RandomPresamplingStrategy is just a presampling method."
                "Please do not specify the downsampled_batch_ratio"
            )
        self._requires_remote_computation = False
