from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


class UncertaintyDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteUncertaintyDownsamplingStrategy"

    def _build_downsampling_params(self) -> dict:
        config = super()._build_downsampling_params()

        if "score_metric" not in self.downsampling_config:
            raise ValueError(
                "Please specify the metric used to score uncertainty for the datapoints. "
                "Available metrics : LeastConfidence, Entropy, Margin"
                "Use the pipeline parameter score_metric"
            )
        config["score_metric"] = self.downsampling_config["score_metric"]

        config["balance"] = self.downsampling_config.get("balance", False)
        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")

        return config
