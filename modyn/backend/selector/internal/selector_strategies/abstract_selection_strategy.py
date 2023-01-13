from abc import ABC, abstractmethod

from modyn.backend.selector.internal.grpc.grpc_handler import GRPCHandler


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selectors. In order to extend this class
    to perform custom experiments, you should override the _select_new_training_samples
    method, which returns a list of tuples given a training ID and the number of samples 
    requested. The tuples can be of arbitrary length, but the first index should be 
    the key of the sample. 

    Then, the selector object should hold an instance of the strategy in selector._strategy. 

    Args:
        config (dict): the configurations for the selector
        grpc (GRPCHandler): the GRPC handler used for calls to metadata database. 
    """

    def __init__(self, config: dict, grpc: GRPCHandler):
        self._config = config
        self._grpc = grpc

    @abstractmethod
    def _select_new_training_samples(self, training_id: int, training_set_size: int) -> list[tuple[str, ...]]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(tuple(str, ...)): the training sample keys for the newly selected training_set with a variable
                       number of auxiliary data (concrete typing in subclasses defined)
        """
        raise NotImplementedError
