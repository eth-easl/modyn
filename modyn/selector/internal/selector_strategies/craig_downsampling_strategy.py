from modyn.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class CRAIGDownsamplingStrategy(AbstractDownsampleStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteCRAIGDownsampling"
