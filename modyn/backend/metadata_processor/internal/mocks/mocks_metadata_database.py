from typing import Iterable, Optional


class SetRequest:
    def __init__(
        self,
        training_id: Optional[int] = ...,
        keys: Optional[Iterable[str]] = ...,
        scores: Optional[Iterable[float]] = ...,
        seen: Optional[Iterable[bool]] = ...,
        label: Optional[Iterable[int]] = ...,
        data: Optional[Iterable[str]] = ...,
    ) -> None:
        self.data = {
            "training_id": training_id,
            "keys": keys,
            "scores": scores,
            "seen": seen,
            "label": label,
            "data": data,
        }


class SetResponse:
    def __init__(self) -> None:
        pass


class MockMetadataDb:
    """Mocks the functionality of the GRPC server used to access the
    Metadata Database"""

    def __init__(self) -> None:
        pass

    def Set(self, request: SetRequest) -> SetResponse:
        return SetResponse()
