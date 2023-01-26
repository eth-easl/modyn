import numpy as np
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class GDumbStrategy(AbstractSelectionStrategy):
    """
    Implements the GDumb selection policy.
    """

    def select_new_training_samples(self, pipeline_id: int) -> list[tuple[str, float]]:
        """
        For a given pipeline_id and number of samples, request that many samples from the selector.

        Returns:
            List of keys for the samples to be considered, along with a default weight of 1.
        """
        result_samples, result_classes = [], []

        all_samples, all_classes = self._get_all_metadata(pipeline_id)
        classes, counts = np.unique(all_classes, return_counts=True)

        num_classes = classes.shape[0]
        if self.training_set_size_limit > 0:
            training_set_size = min(self.training_set_size_limit, len(all_samples))
        else:
            training_set_size = len(all_samples)
        for clss in range(num_classes):
            num_class_samples = counts[clss]
            rand_indices = np.random.choice(num_class_samples, size=training_set_size // num_classes, replace=False)
            class_indices = np.where(all_classes == classes[clss])[0][rand_indices]
            result_samples.append(np.array(all_samples)[class_indices])
            result_classes.append(np.array(all_classes)[class_indices])
        result_samples = np.concatenate(result_samples)
        return [(sample, 1.0) for sample in result_samples]

    def _get_all_metadata(self, pipeline_id: int) -> tuple[list[str], list[int]]:
        all_metadata = (
            self.database.session.query(Metadata.key, Metadata.label).filter(Metadata.pipeline_id == pipeline_id).all()
        )
        return ([metadata.key for metadata in all_metadata], [metadata.label for metadata in all_metadata])

    def inform_data(self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        self.database.set_metadata(
            keys,
            timestamps,
            [None] * len(keys),
            [False] * len(keys),
            labels,
            [None] * len(keys),
            pipeline_id,
        )
