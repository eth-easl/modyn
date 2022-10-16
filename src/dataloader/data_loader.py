from ensurepip import bootstrap
from kafka import KafkaConsumer
from json import loads
class DataLoader:
    config = None
    consumer = None

    def __init__(self, config: dict):
        self.config = config
        self.consumer = KafkaConsumer(
            self.config['kafka']['topic'], 
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            auto_offstet_reset='earliest',
            value_deserializer=lambda x: loads(x.decode('utf-8'))
            )
        # TODO: check the value_deserializer (17.10.2022)
        # TODO: Test (17.10.2022)

    def run(self):
        for message in self.consumer:
            message.value

            self.offline_preprocessing()

            self.write_to_storage()

            self.update_data_importance_server()

    def offline_preprocessing(self):
        # TODO: Implement
        pass

    def write_to_storage(self):
        # TODO: Write to datastorage (17.10.2022)
        pass

    def update_metadata(self):
        # TODO: Write metadata to data orchestrator (17.10.2022)
        pass
