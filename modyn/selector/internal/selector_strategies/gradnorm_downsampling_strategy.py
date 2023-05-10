from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class GradNormDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"

    def get_downsampling_params(self) -> dict:
        return {"downsampled_batch_size": self.downsampled_batch_size}
