from modyn.selector.internal.selector_strategies.downsampling_strategy import AbstractDownsamplingStrategy


class GradNormDownsamplingStrategy(AbstractDownsamplingStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"

    def get_downsampling_params(self) -> dict:
        return {"downsampled_batch_size": self.downsampled_batch_size}
