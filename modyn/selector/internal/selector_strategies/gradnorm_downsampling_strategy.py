from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class GradNormDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"

    def get_downsampling_params(self) -> dict:
        if self.sample_then_batch:
            params = {
                "downsampled_batch_ratio": self.downsampled_batch_ratio,
                "downsampling_period": self.downsampling_period,
                "maximum_keys_in_memory": self._maximum_keys_in_memory,
            }
        else:
            params = {"downsampled_batch_size": self.downsampled_batch_size}
        params["sample_then_batch"] = self.sample_then_batch
        return params
