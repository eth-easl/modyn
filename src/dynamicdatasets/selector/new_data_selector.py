from selector import Selector


class NewDataSelector(Selector):
    def _create_new_training_set(self, training_set_size: int) -> list():
        sample_keys = []
        for i in range(training_set_size):
            sample_keys.append('key - ' + str(i))
        #sample_keys = new_queue_service.get(count = training_set_size)

        return sample_keys

