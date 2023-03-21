from modyn.backend.selector.internal.selector_strategies.abstract_downsample_strategy import AbstractDownsampleStrategy


class LossDownsample(AbstractDownsampleStrategy):
    def __init__(
        self,
        config: dict,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
        downsampled_batch_size: int,
    ) -> None:
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        assert downsampled_batch_size > 0
        self.downsampled_batch_size = downsampled_batch_size

    def get_downsampling_strategy(self) -> str:
        cmd = (
            f"RemoteLossDownsampler(self._model, {self.downsampled_batch_size}, criterion_func("
            f'**training_info.criterion_dict, reduction="none"))'
        )

        return cmd
