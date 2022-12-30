import numpy as np

from modyn.backend.selector.selector import Selector


class GDumbSelector(Selector):
    """Implements the GDumb selection policy.

    Args:
        Selector (dict): the configuration for the selector.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def _select_new_training_samples(self, training_id: int, num_samples: int) -> list:
        """
        For a given training_id and number of samples, request that many samples from the selector.

        Returns:
            List of keys for the samples to be considered.
        """
        result_samples, result_classes = [], []

        all_samples, all_classes = self._get_all_metadata(training_id)
        classes, inverse, counts = np.unique(all_classes, return_inverse=True, return_counts=True)

        num_classes = classes.shape[0]
        for clss in range(num_classes):
            num_class_samples = counts[clss]
            rand_indices = np.random.choice(num_class_samples, size=num_samples // num_classes, replace=False)
            class_indices = np.where(all_classes == classes[clss])[0][rand_indices]
            result_samples.append(np.array(all_samples)[class_indices])
            result_classes.append(np.array(all_classes)[class_indices])

        return zip(list(np.concatenate(result_samples)), list(np.concatenate(result_classes)))

    def _get_all_metadata(self, training_id: int) -> list[str]:
        query = f"SELECT key, score, seen, label, data FROM metadata_database WHERE training_id = {training_id}"
        keys, scores, seen, labels, data = self.get_samples_by_metadata_query(query)
        return data, labels
