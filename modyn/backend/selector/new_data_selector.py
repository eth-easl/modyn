import grpc

from modyn.backend.selector.selector import Selector
# from modyn.backend.newqueue.newqueue_pb2_grpc import NewQueueStub
# from modyn.backend.newqueue.newqueue_pb2 import GetNextRequest


class NewDataSelector(Selector):
    def __init__(self, config: dict):
        super().__init__(config)

        # Setup connection to the new queue server
        newqueue_channel = grpc.insecure_channel(
            self._config['newqueue']['hostname'] +
            ':' +
            self._config['newqueue']['port'])
        # self.__newqueue_stub = NewQueueStub(newqueue_channel)

    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list:
        # request = GetNextRequest(
        #     limit=training_set_size,
        #     training_id=training_id)
        # sample_keys = self.__newqueue_stub.GetNext(request).keys
        # return sample_keys
        return []
