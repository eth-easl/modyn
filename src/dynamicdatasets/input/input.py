import time

import grpc

from dynamicdatasets.preprocess.preprocess_pb2_grpc import PreprocessStub


class Input:

    def __init__(self, config: dict) -> None:
        self.__config = config

        self.adapter_module = self.my_import('dynamicdatasets.input.adapter.')
        self.__adapter = getattr(
            self.adapter_module,
            config['input']['adapter'])(config)

        preprocess_channel = grpc.insecure_channel(
            self.__config['newqueue']['hostname'] +
            ':' +
            self.__config['newqueue']['port'])
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
                self.__preprocess_stub.Preprocess(data)
            time.sleep(1)


if __name__ == '__main__':
    import logging
    import sys
    import yaml

    logging.basicConfig()
    with open(sys.argv[1], 'r') as stream:
        config = yaml.safe_load(stream)
    input = Input(config)
    input.run()
