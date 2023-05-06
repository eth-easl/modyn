from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class RandomDownsamplingStrategy(AbstractDownsampleStrategy):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        assert "downsampled_batch_size" not in config, "This strategy just performs presampling. Omit this parameter"
        config["downsampled_batch_size"] = -1

        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self._requires_remote_computation = False
