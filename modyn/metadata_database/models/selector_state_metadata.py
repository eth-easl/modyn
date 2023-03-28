"""SelectorStateMetadata model."""

import logging

from modyn.metadata_database.metadata_base import MetadataBase, PartitionByMeta
from sqlalchemy import BigInteger, Boolean, Column, Index, Integer, inspect
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import text

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")

logger = logging.getLogger(__name__)


class SelectorStateMetadataMixin:
    pipeline_id = Column("pipeline_id", Integer, primary_key=True)
    sample_key = Column("sample_key", BIGINT, primary_key=True)
    seen_in_trigger_id = Column("seen_in_trigger_id", Integer, primary_key=True)
    used = Column("used", Boolean, default=False)
    timestamp = Column("timestamp", BigInteger)
    label = Column("label", Integer)


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
    indexes = {"ssm_pipeline_seen_idx": ["pipeline_id", "seen_in_trigger_id"]}

    __table_args__ = (*[Index(index[0], *index[1]) for index in indexes.items()],)

    @staticmethod
    def add_pipeline(pipeline_id: int) -> PartitionByMeta:
        partition_stmt = f"FOR VALUES IN ({pipeline_id})"
        partition_suffix = f"_pid{pipeline_id}"
        result = SelectorStateMetadata.create_partition(
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by="seen_in_trigger_id",
            subpartition_type="LIST",
        )
        return result

    @staticmethod
    def add_trigger(
        pipeline_id: int, trigger_id: int, session: Session, engine: Engine, hash_partition_modulus: int = 16
    ) -> None:
        #  If sqlite, do not partition
        if session.bind.dialect.name == "sqlite":
            return
        logger.debug(f"Creating partition for trigger {trigger_id} in pipeline {pipeline_id}")
        #  Create partition for pipeline
        pipeline_partition = SelectorStateMetadata.add_pipeline(pipeline_id)
        SelectorStateMetadata.create_partition_table(pipeline_partition, engine)
        # Create partition for trigger
        partition_suffix = f"_tid{trigger_id}"
        partition_stmt = f"FOR VALUES IN ({trigger_id})"
        trigger_partition = pipeline_partition.create_partition(
            partition_suffix, partition_stmt=partition_stmt, subpartition_by="sample_key", subpartition_type="HASH"
        )
        SelectorStateMetadata.create_partition_table(trigger_partition, engine)

        # Create partitions for sample key hash
        session.execute(text("SET enable_parallel_hash=off;"))
        try:
            for i in range(hash_partition_modulus):
                partition_suffix = f"_part{i}"
                partition_stmt = f"FOR VALUES WITH (modulus {hash_partition_modulus}, remainder {i})"
                modulo_partition = trigger_partition.create_partition(partition_suffix, partition_stmt=partition_stmt)
                SelectorStateMetadata.create_partition_table(modulo_partition, engine)
            logger.debug(
                f"Created {hash_partition_modulus} hash partitions for trigger {trigger_id} in pipeline {pipeline_id}"
            )
        finally:
            session.execute(text("SET enable_parallel_hash=on;"))

    @staticmethod
    def create_partition_table(table: PartitionByMeta, engine: Engine) -> None:
        """Create partition table if not exists."""
        if not inspect(engine).has_table(table.__tablename__):
            table.__table__.create(engine)

    @staticmethod
    def disable_indexes(engine: Engine) -> None:
        """Disable indexes for faster inserts."""
        if engine.dialect.name == "sqlite":
            return
        for index in SelectorStateMetadata.indexes:
            with engine.connect() as conn:
                with conn.execution_options(isolation_level="AUTOCOMMIT"):
                    conn.execute(text(f"DROP INDEX IF EXISTS {index};"))

    @staticmethod
    def enable_indexes(engine: Engine) -> None:
        """Enable indexes after inserts."""
        if engine.dialect.name == "sqlite":
            return
        for index_name, index_items in SelectorStateMetadata.items():
            with engine.connect() as conn:
                with conn.execution_options(isolation_level="AUTOCOMMIT"):
                    conn.execute(
                        text(
                            f"CREATE INDEX {index_name} ON {SelectorStateMetadata.__tablename__} \
                            ({', '.join(index_items)});"
                        )
                    )
