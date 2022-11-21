from selector_pb2_grpc import SelectorStub
from selector_pb2 import *

import grpc


channel = grpc.insecure_channel('127.0.0.1:5444')
stub = SelectorStub(channel=channel)


#tid = stub.register_training(RegisterTrainingRequest(training_set_size=8, num_workers=2))
#print(tid)

samples = stub.get_sample_keys(GetSamplesRequest(training_id=2, training_set_number=1, worker_id=0))
print(samples)



