import numpy as np
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class ScoreStrategy(AbstractSelectionStrategy):
    """Implements a score based selector. Has two modes: softmax mode and normal mode.
    As the name suggests, softmax mode will apply softmax to the samples (so the scores
    can be negative). Defaults to softmax mode.

    Args:
        config (dict): configuration for the selector
    """

    def __init__(self, config: dict, modyn_config: dict):
        super().__init__(config, modyn_config)
        self._set_is_softmax_mode(config["selector"].get("softmax_mode", True))

    def _set_is_softmax_mode(self, is_softmax_mode: bool) -> None:
        self.is_softmax_mode = is_softmax_mode

    def select_new_training_samples(self, training_id: int) -> list[tuple[str, float]]:
        """
        For a given training_id and number of samples, request that many samples from
        the selector.

        Returns:
            List of keys for the samples that we want to select.
        """
        all_samples, all_scores = self._get_all_metadata(training_id)
        all_samples_np = np.array(all_samples)
        all_scores_np = np.array(all_scores)

        if self.training_set_size_limit > 0:
            training_set_size = min(self.training_set_size_limit, len(all_samples))
        else:
            training_set_size = len(all_samples)

        if self.is_softmax_mode:
            all_scores_np = np.exp(all_scores_np) / np.sum(np.exp(all_scores_np))
        else:
            assert all_scores_np.min() >= 0, "Scores should be nonnegative if on normal mode!"
            all_scores_np = all_scores_np / np.sum(all_scores_np)
        rand_indices = np.random.choice(
            all_samples_np.shape[0],
            size=training_set_size,
            replace=False,
            p=all_scores_np,
        )
        samples = all_samples_np[rand_indices]
        scores = all_scores_np[rand_indices]
        return list(zip(list(samples), list(scores)))

    def _get_all_metadata(self, training_id: int) -> tuple[list[str], list[float]]:
        all_metadata = self.database.session.query(Metadata).filter(Metadata.training_id == training_id).all()
        return ([metadata.key for metadata in all_metadata], [metadata.score for metadata in all_metadata])

    def inform_data(self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        self.database.set_metadata(
            keys,
            timestamps,
            [1.0] * len(keys),
            [False] * len(keys),
            [None] * len(keys),
            [None] * len(keys),
            pipeline_id,
        )

    def trigger(self) -> None:
        # For score strategy, no work to do.
        pass
