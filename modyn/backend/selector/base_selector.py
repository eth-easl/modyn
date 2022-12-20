import grpc

from modyn.backend.selector.selector import Selector


class BaseSelector(Selector):
    def __init__(self, config: dict):
        super().__init__(config)

        self._set_new_data_ratio(self._config['selector']['new_data_ratio'])
        self._set_is_adaptive_ratio(self._config['selector']['is_adaptive_ratio'])

        # Setup connection to the new queue server
        # newqueue_channel = grpc.insecure_channel(
        #     self._config['newqueue']['hostname'] +
        #     ':' +
        #     self._config['newqueue']['port'])
        # self.__newqueue_stub = NewQueueStub(newqueue_channel)

    def _set_new_data_ratio(self, new_data_ratio) -> None:
        assert new_data_ratio >= 0 and new_data_ratio <= 1
        self.new_data_ratio = new_data_ratio
        self.old_data_ratio = 1 - self.new_data_ratio

    def _set_is_adaptive_ratio(self, is_adaptive_ratio) -> None:
        self._is_adaptive_ratio = is_adaptive_ratio

    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list:
        if self._is_adaptive_ratio:
            newqueue_size = self.get_newqueue_size(training_id)
            self.new_data_ratio = newqueue_size / (self.get_odm_size + newqueue_size)

        num_new_samples = int(training_set_size * self.new_data_ratio)
        num_old_samples = training_set_size - num_new_samples
        new_samples = self.get_from_newqueue(training_id, num_new_samples)
        old_samples = self.get_from_odm(training_id, num_old_samples)
        new_samples.extend(old_samples)
        return new_samples
        # request = GetNextRequest(
        #     limit=training_set_size,
        #     training_id=training_id)
        # sample_keys = self.__newqueue_stub.GetNext(request).keys
        # return sample_keys
