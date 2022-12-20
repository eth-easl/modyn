import grpc

import numpy as np

from modyn.backend.selector.base_selector import BaseSelector


class GDumbSelector(BaseSelector):

    def __init__(self, config: dict):
        super().__init__(config)

        self._set_new_data_ratio(0.0)
        self._set_is_adaptive_ratio(False)

    def get_from_newqueue(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the new queue.

        Returns:
            List of keys for the samples in the new queue.
        """
        # GDumb as a strategy does not take samples from new queue.
        return []

    def get_from_odm(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the ODM service.

        Returns:
            List of keys for the samples in the ODM.
        """
        result_samples, result_classes = [], []

        all_odm_samples, all_odm_classes = self._get_all_odm(training_id)
        classes, inverse, counts = np.unique(all_odm_classes, return_inverse=True, return_counts=True)

        num_classes = classes.shape[0]
        for clss in range(num_classes):
            num_class_samples = counts[clss]
            rand_indices = np.random.choice(num_class_samples, size=num_samples // num_classes, replace=False)
            class_indices = np.where(all_odm_classes == classes[clss])[0][rand_indices]
            result_samples.append(np.array(all_odm_samples)[class_indices])
            result_classes.append(np.array(all_odm_classes)[class_indices])

        return zip(list(np.concatenate(result_samples)), list(np.concatenate(result_classes)))

    def _get_all_odm(self, training_id: int) -> list[str]:
        raise NotImplementedError
