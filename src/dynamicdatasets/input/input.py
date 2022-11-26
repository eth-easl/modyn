import time

import grpc

from dynamicdatasets.preprocess.preprocess_pb2_grpc import PreprocessStub
from dynamicdatasets.preprocess.preprocess_pb2 import PreprocessRequest


class Input:

    def __init__(self, config: dict) -> None:
        self.__config = config

        self.adapter_module = self.my_import('dynamicdatasets.input.adapter')
        self.__adapter = getattr(
            self.adapter_module,
            config['input']['adapter'])(config)

        preprocess_channel = grpc.insecure_channel(
            self.__config['preprocess']['hostname'] +
            ':' +
            self.__config['preprocess']['port'])
        self.__preprocess_stub = PreprocessStub(preprocess_channel)

    def my_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def run(self):
        while True:
            data = self.__adapter.get_next()

            if data is not None:
                print("Sending data to preprocess")
                self.__preprocess_stub.Preprocess(
                    PreprocessRequest(value=data))
            time.sleep(self.__config['input']['send_batch_interval'])


if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python input.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    input = Input(config)
    input.run()
