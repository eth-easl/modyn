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

    def _select_new_training_samples(self, training_id: int, num_samples: int) -> list:
        """
        For a given training_id and number of samples, request that many samples from
        the ODM service.

        Returns:
            List of keys for the samples in the ODM.
        """
        result_samples, result_scores = [], []

        all_samples, all_scores = self._get_all_metadata(training_id)
        all_samples = np.array(all_samples)
        all_scores = np.array(all_scores)
        if self.is_softmax_mode:
            all_scores = np.exp(all_scores) / np.sum(np.exp(all_scores))
        else:
            assert all_scores.min() >= 0, "Scores should be nonnegative if on normal mode!"
            all_scores = all_scores / np.sum(all_scores)
        rand_indices = np.random.choice(all_samples.shape[0], size=num_samples, replace=False, p = all_scores)
        samples = all_samples[rand_indices]
        scores = all_scores[rand_indices]
        return zip(list(samples), list(scores))

    def _get_all_metadata(self, training_id: int) -> list[str]:
        query = f"SELECT key, score, seen, label, data FROM odm_storage WHERE training_id = {training_id}"
        keys, scores, seen, labels, data = self.get_samples_by_metadata_query(query)
        return data, scores