from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class LossDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteLossDownsampling"

    def get_downsampling_params(self) -> dict:
        if self.sample_before_batch:
            params = {
                "downsampled_batch_ratio": self.downsampled_batch_ratio,
                "downsampling_period": self.downsampling_period,
                "maximum_keys_in_memory": self._maximum_keys_in_memory,
            }
        else:
            params = {"downsampled_batch_size": self.downsampled_batch_size}
        params["sample_before_batch"] = self.sample_before_batch
        return params
