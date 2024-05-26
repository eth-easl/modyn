from typing import Optional, Any

from sqlalchemy.orm import Session

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend


def get_trigger_dataset_size(
        storage_backend: AbstractStorageBackend,
        pipeline_id: int,
        trigger_id: int,
        tail_triggers: Optional[int]
) -> int:
    # Count the number of samples that might be sampled during the next trigger. Typically used to compute the
    # target size for presampling_strategies (target_size = trigger_dataset_size * ratio)
    assert isinstance(
        storage_backend, DatabaseStorageBackend
    ), "CoresetStrategy currently only supports DatabaseBackend"

    def _session_callback(session: Session) -> Any:
        return (
            session.query(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == pipeline_id,
                (
                    SelectorStateMetadata.seen_in_trigger_id >= trigger_id - tail_triggers
                    if tail_triggers is not None
                    else True
                ),
                )
            .count()
        )

    return storage_backend._execute_on_session(_session_callback)
