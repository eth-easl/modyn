"""SelectorStateMetadata model."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import BigInteger, Boolean, Column, Integer
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session

from modyn.database import PartitionByMeta
from modyn.metadata_database.metadata_base import MetadataBase

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")

logger = logging.getLogger(__name__)


class SelectorStateMetadataMixin:
    pipeline_id = Column("pipeline_id", Integer, primary_key=True)
    sample_key = Column("sample_key", BIGINT, primary_key=True)
    seen_in_trigger_id = Column("seen_in_trigger_id", Integer, primary_key=True)
    used = Column("used", Boolean, default=False)
    # This is a field to help selection strategies to track the state of the samples during preparing the
    # post-presampling training set. It should be reset to 0 after the training set is prepared.
    # For example, the RHOLossDownsamplingStrategy uses this field to split the training set into two halves
    # by first sampling 50% and setting their `tmp_version` field to 1,
    # Then it generates the remaining 50% by querying samples with `tmp_version` as 0.
    tmp_version = Column("tmp_version", Integer, default=0)
    timestamp = Column("timestamp", BigInteger)
    label = Column("label", Integer)
    last_used_in_trigger = Column("last_used_in_trigger", Integer, default=-1)


# 1. Partition level: pipeline_id (LIST, manually create partition on new pipeline)
# 2. Parititon level: seen_in_trigger_id (LIST, manually create partition on trigger for next trigger)
# 3. Partition level: sample_key (HASH, MOD 16/32/64)
class SelectorStateMetadata(
    SelectorStateMetadataMixin,
    MetadataBase,
    metaclass=PartitionByMeta,
    partition_by="pipeline_id",  # type: ignore
    partition_type="LIST",  # type: ignore
):
    """SelectorStateMetadata model.

    Metadata persistently stored by the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "selector_state_metadata"

    @staticmethod
    def add_pipeline(pipeline_id: int, session: Session, engine: Engine) -> PartitionByMeta | None:
        partition_stmt = f"FOR VALUES IN ({pipeline_id})"
        partition_suffix = f"_pid{pipeline_id}"
        result = SelectorStateMetadata._create_partition(
            SelectorStateMetadata,
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by="seen_in_trigger_id",
            subpartition_type="LIST",
            session=session,
            engine=engine,
        )
        return result

    @staticmethod
    def add_trigger(
        pipeline_id: int, trigger_id: int, session: Session, engine: Engine, hash_partition_modulus: int = 16
    ) -> None:
        logger.debug(f"Creating partition for trigger {trigger_id} in pipeline {pipeline_id}")
        #  Create partition for pipeline
        pipeline_partition = SelectorStateMetadata.add_pipeline(pipeline_id, session, engine)
        if pipeline_partition is None:
            # Happens when partitioning is disabled, e.g., in sqlite instead of postgres
            return
        # Create partition for trigger
        partition_suffix = f"_tid{trigger_id}"
        partition_stmt = f"FOR VALUES IN ({trigger_id})"
        trigger_partition = SelectorStateMetadata._create_partition(
            pipeline_partition,
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by="sample_key",
            subpartition_type="HASH",
            session=session,
            engine=engine,
        )

        # Create partitions for sample key hash
        for i in range(hash_partition_modulus):
            partition_suffix = f"_part{i}"
            partition_stmt = f"FOR VALUES WITH (modulus {hash_partition_modulus}, remainder {i})"
            _ = SelectorStateMetadata._create_partition(
                trigger_partition,
                partition_suffix,
                partition_stmt=partition_stmt,
                subpartition_by=None,
                subpartition_type=None,
                session=session,
                engine=engine,
            )

    @staticmethod
    def _create_partition(
        instance: Any,  # This is the class itself
        partition_suffix: str,
        partition_stmt: str,
        subpartition_by: str | None,
        subpartition_type: str | None,
        session: Session,
        engine: Engine,
    ) -> PartitionByMeta | None:
        """Create a partition for the SelectorStateMetadata table."""
        #  If sqlite, do not partition
        if session.bind.dialect.name == "sqlite":
            return None

        #  Create partition
        partition = instance.create_partition(
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by=subpartition_by,
            subpartition_type=subpartition_type,
        )

        #  Create table
        SelectorStateMetadata.metadata.create_all(engine, [partition.__table__])

        return partition
