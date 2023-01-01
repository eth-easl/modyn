from abc import ABC, abstractmethod
from modyn.backend.selector.internal.grpc_handler import GRPCHandler


class Selector(ABC):
    """This class is the base class for selectors. In order to extend this class
    to perform custom experiments, the most important thing is to implement the
    _select_new_training_samples method, which should return a list of keys given
    a training ID and the number of samples requested. To do so, make use of
    get_samples_by_metadata_query, which will get samples from the metadata service
    using a query.

    Args:
        Selector (config): the configurations for the selector


    """

    def __init__(self, config: dict):
        self.grpc = GRPCHandler(config)
        self._config = config
    
    @abstractmethod
    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list[str]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(str): the training sample keys for the newly selected training_set
        """
        raise NotImplementedError

    def _prepare_training_set(
            self,
            training_id: int,
            training_set_number: int,
            training_set_size: int,
    ) -> list[str]:
        """
        Create a new training set of samples for the given training id. New samples are selected from
        the select_new_samples method and are inserted into the database for the given set number.

        Returns:
            list(str): the training sample keys for the newly prepared training_set
        """
        training_samples = self._select_new_training_samples(training_id, training_set_size)

        # Throw error if no new samples are selected
        if (len(training_samples) == 0):
            raise ValueError("No new samples selected")

        return training_samples

    def _get_training_set_partition(
            self,
            training_id: int,
            training_samples: list[str],
            worker_id: int) -> list[str]:
        """
        Return the required subset of training samples for the particular worker id
        The subset is calculated by taking an offset from the start based on the given worker id
        """
        training_set_size, num_workers = self._get_info_for_training(training_id)

        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f'Asked for worker id {worker_id}, but only have {num_workers} workers!')

        worker_subset_size = int(training_set_size / num_workers)
        if training_set_size % num_workers > 0:
            worker_subset_size += 1
        start_index = worker_id * worker_subset_size
        training_samples_subset = training_samples[start_index: min(start_index +
                                                   worker_subset_size, len(training_samples))]
        return training_samples_subset

    def register_training(self, training_set_size: int,
                          num_workers: int) -> int:
        """
        Creates a new training object in the database with the given training_set_size and num_workers
        Returns:
            The id of the newly created training object
        Throws:
            ValueError if training_set_size or num_workers is not positive.
        """
        if num_workers <= 0 or training_set_size <= 0:
            raise ValueError(
                f'Tried to register training with {num_workers} workers and {training_set_size} data points.')

        return self._register_training(training_set_size, num_workers)

    def _register_training(self, training_set_size: int,
                           num_workers: int) -> int:
        """
        Creates a new training object in the database with the given training_set_size and num_workers
        Returns:
            The id of the newly created training object
        """
        return self.grpc.register_training(training_set_size, num_workers)

    def _get_info_for_training(self, training_id: int) -> tuple[int, int]:
        """
        Queries the database for the the training set size and number of workers for a given training.

        Returns:
            Tuple of training set size and number of workers.
        """
        return self.grpc.get_info_for_training(training_id)

    def get_sample_keys(self, training_id: int,
                        training_set_number: int, worker_id: int) -> list[str]:
        """
        For a given training_id, training_set_number and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage.

        Returns:
            List of keys for the samples to be returned to that particular worker
        """
        training_set_size, num_workers = self._get_info_for_training(training_id)
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(f'Training {training_id} has {num_workers} workers, but queried for worker {worker_id}!')

        training_samples = self._prepare_training_set(training_id, training_set_number, training_set_size)

        training_samples_subset = self._get_training_set_partition(
            training_id, training_samples, worker_id)

        return training_samples_subset

    def get_samples_by_metadata_query(
            self, query: str) -> tuple[list[str], list[float], list[bool], list[int], list[str]]:
        return self.grpc.get_samples_by_metadata_query(query)
