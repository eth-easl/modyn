class OnlineDataLoader:
    config = None

    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        # TODO: Load data from data storage node
        pass

    def get_next_batch(self):
        # TODO: Get what data to feed from data orchestrator
        pass

    def get_next(self):
        # TODO: Connect instance with trainer to feed data fast and efficiently
        pass