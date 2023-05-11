from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class GradNormDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"

    def get_downsampling_params(self) -> dict:
        if self.sample_before_batch:
            params = {"downsampled_batch_ratio": self.downsampled_batch_ratio}
        else:
            params = {"downsampled_batch_size": self.downsampled_batch_size}
        params["sample_before_batch"] = self.sample_before_batch
        return params
