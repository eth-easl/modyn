from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


class LossDownsamplingStrategy(AbstractDownsamplingStrategy):
    def get_downsampling_strategy(self) -> str:
        return "RemoteLossDownsampling"

    def get_downsampling_params(self) -> dict:
        return {"downsampled_batch_size": self.downsampled_batch_size}
