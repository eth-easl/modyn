from modyn.backend.selector.selector import Selector
import numpy as np


class ScoreSelector(Selector):
    """Implements a score based selector. Has two modes: softmax mode and normal mode.
    As the name suggests, softmax mode will apply softmax to the samples (so the scores
    can be negative). Defaults to softmax mode.

    Args:
        Selector (dict): configuration for the selector
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._set_is_softmax_mode(config['selector'].get('softmax_mode', True))

    def _set_is_softmax_mode(self, is_softmax_mode: bool) -> None:
        self.is_softmax_mode = is_softmax_mode

    def _select_new_training_samples(self, training_id: int, num_samples: int) -> list[tuple[str, float]]:
        """
        For a given training_id and number of samples, request that many samples from
        the selector.

        Returns:
            List of keys for the samples in the ODM.
        """
        all_samples, all_scores = self._get_all_metadata(training_id)
        all_samples_np = np.array(all_samples)
        all_scores_np = np.array(all_scores)

        if self.is_softmax_mode:
            all_scores_np = np.exp(all_scores_np) / np.sum(np.exp(all_scores_np))
        else:
            assert all_scores_np.min() >= 0, "Scores should be nonnegative if on normal mode!"
            all_scores_np = all_scores_np / np.sum(all_scores_np)
        rand_indices = np.random.choice(all_samples_np.shape[0], size=num_samples, replace=False, p=all_scores_np)
        samples = all_samples_np[rand_indices]
        scores = all_scores_np[rand_indices]
        return list(zip(list(samples), list(scores)))

    def _get_all_metadata(self, training_id: int) -> tuple[list[str], list[float]]:
        query = f"SELECT key, score, seen, label, data FROM metadata_database WHERE training_id = {training_id}"
        keys, scores, seen, labels, data = self.get_samples_by_metadata_query(query)
        return data, scores
