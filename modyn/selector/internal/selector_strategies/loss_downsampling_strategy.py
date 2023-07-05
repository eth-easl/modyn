from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class LossDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteLossDownsampling"
