from abc import ABC, abstractmethod


class AbstractDownsamplingStrategy(ABC):

    """
    This abstract strategy is used to represent the common behaviour of downsampling strategies
    like loss-based, importance downsampling (distribution-based methods) and craig&adacore (greedy-based methods)

    These methods work on a uniformly-presampled version of the entire dataset (relying on AbstractPresampleStrategy),
    then the actual downsampling is done at the trainer server since all of these methods rely on the result of
    forward pass.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict):
        if "downsampled_batch_size" not in config:
            raise ValueError("To use downsampling strategies, you have to specify the downsampled_batch_size")

        self.downsampled_batch_size = config["downsampled_batch_size"]

        if not isinstance(self.downsampled_batch_size, int):
            raise ValueError("The downsampled batch size must be an integer")

        self.requires_remote_computation = True

    def get_requires_remote_computation(self) -> bool:
        return self.requires_remote_computation

    @abstractmethod
    def get_downsampling_strategy(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_downsampling_params(self) -> dict:
        raise NotImplementedError
