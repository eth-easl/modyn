from backend.selector import Selector


class ScoreSelector(Selector):
    def _create_new_training_set(self, training_set_size: int) -> list():
        sample_keys = []
        for i in range(training_set_size):
            sample_keys.append('key - ' + str(i))
        # sample_keys = odm_service.get(query = "select key from <> order by score desc")

        return sample_keys
