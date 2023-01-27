from abc import ABC


class BaseCallback(ABC):
    def __init__(self) -> None:
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_batch_begin(self) -> None:
        pass

    def on_batch_before_update(self) -> None:
        pass

    def on_batch_end(self) -> None:
        pass
