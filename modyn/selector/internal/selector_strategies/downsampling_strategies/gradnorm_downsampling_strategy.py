from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class GradNormDownsamplingStrategy(AbstractDownsamplingStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteGradNormDownsampling"
