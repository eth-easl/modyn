from abc import ABC


class BaseCallback(ABC):
    def __init__(self) -> None:
        super().__init__()

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass