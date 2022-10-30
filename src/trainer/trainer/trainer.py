from onlinedataloader.onlinedataloader.online_data_loader import OnlineDataLoader


from onlinedataloader import OnlineDataLoader


class Trainer:
    __config = None
    __online_data_loader = None

    def __init__(self, config: dict):
        self.__config = config
        self.__online_data_loader = OnlineDataLoader(config)

    def train(self):
        # TODO: Implement
        pass

    def fit(self):
        # TODO: Implement
        pass

    def online_training(self):
        # TODO: Implement
        pass

    def run(self):
        # TODO: Implement
        pass
