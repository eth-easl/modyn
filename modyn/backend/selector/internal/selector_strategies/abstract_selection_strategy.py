from abc import ABC, abstractmethod


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selectors. In order to extend this class
    to perform custom experiments, you should override the _select_new_training_samples
    method, which returns a list of tuples given a training ID and the number of samples
    requested. The tuples can be of arbitrary length, but the first index should be
    the key of the sample.

    Then, the selector object should hold an instance of the strategy in selector._strategy.

    Args:
        config (dict): the configurations for the selector
        modyn_config (dict): the configurations for the modyn backend
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int):
        self._config = config
        assert (
            "limit" in config.keys() and "reset_after_trigger" in config.keys()
        ), "Strategy instantiated with invalid config"

        self.training_set_size_limit: int = config["limit"]
        self.reset_after_trigger: bool = config["reset_after_trigger"]
        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id

    @abstractmethod
    def _on_trigger(self) -> list[tuple[str, float]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on.

        Returns:
            list(tuple(str, float)): Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_state(self) -> None:
        """Resets the internal state of the strategy, e.g., by clearing buffers."""
        raise NotImplementedError

    @abstractmethod
    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        """Informs the strategy of new data.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
        """
        raise NotImplementedError

    def trigger(self) -> list[tuple[str, float]]:
        """
        Causes the strategy to compute the training set, and (if so configured) reset its internal state.

        Returns:
            list(tuple(str, float)): each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        training_samples = self._on_trigger()

        if self.reset_after_trigger:
            self._reset_state()

        return training_samples
