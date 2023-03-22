"""SelectorStateMetadata model."""

from typing import Any

from modyn.metadata_database.metadata_base import PartitionByMeta, MetadataBase
from sqlalchemy import BigInteger, Boolean, Column, Index, Integer
from sqlalchemy.dialects import sqlite

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class SelectorStateMetadataMixin:
    pipeline_id = Column("pipeline_id", Integer, primary_key=True)
    sample_key = Column("sample_key", BIGINT, primary_key=True)
    seen_in_trigger_id = Column("seen_in_trigger_id", Integer)
    used = Column("used", Boolean, default=False)
    timestamp = Column("timestamp", BigInteger)
    label = Column("label", Integer)


# 1. Partition level: pipeline_id (LIST, manually create partition on new pipeline)
# 2. Parititon level: seen_in_trigger_id (LIST, manually create partition on trigger for next trigger)
# 3. Partition level: sample_key (HASH, MOD 16/32/64)
# TODO: Do not partition for sqllite
class SelectorStateMetadata(
    SelectorStateMetadataMixin,
    MetadataBase,
    metaclass=PartitionByMeta,
    partition_by="pipeline_id",
    partition_type="LIST",
):
    """SelectorStateMetadata model.

    Metadata persistently stored by the Selector.

    Additional required columns need to be added here.
    """

    __tablename__ = "selector_state_metadata"
    indexes = {"ssm_pipeline_seen_idx": ["pipeline_id", "seen_in_trigger_id"]}

    # ssm_pipeline_seen_idx: Optimizes new data strategy
    # (Index("ssm_pipeline_seen_idx", "pipeline_id", "seen_in_trigger_id")
    # TODO: integrate index again on the fly
    #  Create index non-blocking
    __table_args__ = (*[Index(index[0], *index[1]) for index in indexes.items()],)

    @staticmethod
    def add_pipeline(pipeline_id: int) -> Any:  # TODO investigate type
        partition_stmt = f"FOR VALUES IN ({pipeline_id})"
        partition_suffix = f"_pid{pipeline_id}"
        result = SelectorStateMetadata.create_partition(
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by="seen_in_trigger_id",
            subpartition_type="LIST",
        )
        print(f"Created partition {result.__tablename__} for pipeline {pipeline_id}")
        print(f"Type: {type(result)}")
        return result

    @staticmethod
    def add_trigger(pipeline_id: int, trigger_id: int) -> None:
        # TODO: this is not the best/correct way to obtain the partition.
        # We should somehow implement a "get partition"
        # (and populate all partitions on boot and let add_pipeline fail if partition already exists!)
        #  Where are the partitions and are they loaded upon restart?
        pipeline_partition = SelectorStateMetadata.add_pipeline(pipeline_id)
        partition_suffix = f"_tid{trigger_id}"
        partition_stmt = f"FOR VALUES IN ({trigger_id})"
        trigger_partition = pipeline_partition.create_partition(
            partition_suffix, partition_stmt=partition_stmt, subpartition_by="sample_key", subpartition_type="HASH"
        )

        # Create hash subpartitions
        modulus = 16  # make configurable in config

        for i in range(modulus):
            partition_suffix = f"_part{i}"
            partition_stmt = f"FOR VALUES WITH (modulus {modulus}, remainder {i})"
            trigger_partition.create_partition(partition_suffix, partition_stmt=partition_stmt)
