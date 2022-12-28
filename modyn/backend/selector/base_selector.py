
from modyn.backend.selector.selector import Selector
import numpy as np

class BasicSelector(Selector):
    """
    This class implements selection solely based on freshness of the data. 
    Specifically, there is a "unseen_data_ratio" that controls 
    how much of each batch is from unseen data, and how much is from previously
    seen data. If is_adaptive_ratio is set to True, then this ratio is automatically
    set to the proportion of the size of the unseen vs. previously sen data. 

    Args:
        Selector (dict): The configuration for the selector. 
    """
    def __init__(self, config: dict):
        super().__init__(config)

        self._set_unseen_data_ratio(self._config['selector']['unseen_data_ratio'])
        self._set_is_adaptive_ratio(self._config['selector']['is_adaptive_ratio'])

    def _set_unseen_data_ratio(self, unseen_data_ratio: float) -> None:
        assert unseen_data_ratio >= 0 and unseen_data_ratio <= 1
        self.unseen_data_ratio = unseen_data_ratio
        self.old_data_ratio = 1 - self.unseen_data_ratio

    def _set_is_adaptive_ratio(self, is_adaptive_ratio: bool) -> None:
        self._is_adaptive_ratio = is_adaptive_ratio

    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list:
        if self._is_adaptive_ratio:
            seen_data_size = self.get_seen_data_size(training_id)
            unseen_data_size = self.get_unseen_data_size(training_id)
            self.unseen_data_ratio = unseen_data_size / (unseen_data_size + seen_data_size)

        num_new_samples = int(training_set_size * self.unseen_data_ratio)
        num_old_samples = training_set_size - num_new_samples
        new_samples = self.get_unseen_data(training_id, num_new_samples)
        old_samples = self.get_seen_data(training_id, num_old_samples)
        new_samples.extend(old_samples)
        return new_samples


    def get_unseen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the new queue.

        Returns:
            List of keys for the samples in the new queue.
        """
        query = f"SELECT key, score, seen, label, data FROM odm_storage WHERE seen = 0 AND training_id = {training_id}"
        keys, scores, seen, label, data = self.get_samples_by_metadata_query(query)
        assert not np.array(seen).any()  
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return keys[choice]


    def get_seen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the ODM service.

        Returns:
            List of keys for the samples in the ODM.
        """
        query = f"SELECT key, score, seen, label, data FROM odm_storage WHERE seen = 1 AND training_id = {training_id}"
        keys, scores, seen, label, data = self.get_samples_by_metadata_query(query)
        assert np.array(seen).all()  
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return keys[choice]

    def get_seen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many samples are in the new queue.

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of samples in the new queue.
        """
        query = f"SELECT key, score, seen, label, data FROM odm_storage WHERE seen = 0 AND training_id = {training_id}"
        keys, scores, seen, label, data = self.get_samples_by_metadata_query(query)
        assert not np.array(seen).any()  
        return len(keys)

    def get_unseen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many samples are in the ODM.

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of samples in the ODM
        """
        query = f"SELECT key, score, seen, label, data FROM odm_storage WHERE seen = 1 AND training_id = {training_id}"
        keys, scores, seen, data = self.get_samples_by_metadata_query(query)
        assert np.array(seen).all()  
        return len(keys)