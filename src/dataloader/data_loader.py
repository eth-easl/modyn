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
        # TODO: check the value_deserializer

    def run(self):
        for message in self.consumer:
            message.value

        # TODO: What should we do with a message
