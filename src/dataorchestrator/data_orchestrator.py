import pathlib
import json

STORAGE_LOCATION = pathlib.Path(__file__).parent.resolve() + '/store'

class DataOrchestrator:
    config = None

    def __init__(self, config: dict):
        self.config = config

    def add_to_metadata(self, metadata: dict):
        with open(f'{STORAGE_LOCATION}/metadata.json', 'a+') as f:
            metadata_object = json.load(f)
            metadata_object.setdefault('files', [])
            metadata_object['files'] += metadata

        with open(f'{STORAGE_LOCATION}/metadata.json', 'w') as f:
            json.dump(metadata_object, f)

    def update_metadata(self):
        # TODO: Replace metadata with a relational database to ensure ACID (17.10.2022)
        # TODO: Keep track of the data in storage and its importance metrics (17.10.2022)
        pass

    def update_batches(self):
        # TODO: Read, write and reshuffle data in storage
        pass

    def run(self):
        # TODO: Have a pool of online data loaders ready to feed the trainers
        # TODO: Decide on what data to feed when and where
        pass