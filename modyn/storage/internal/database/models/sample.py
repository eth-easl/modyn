"""Sample model."""

from typing import Any, Optional

from modyn.database import PartitionByMeta
from modyn.storage.internal.database.storage_base import StorageBase
from sqlalchemy import BigInteger, Column, ForeignKey, Integer
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import Session

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class SampleMixin:
    sample_id = Column("sample_id", BIGINT, autoincrement=True, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"))
    file_id = Column(Integer, ForeignKey("files.file_id"), index=True)
    index = Column(BigInteger, nullable=False)
    label = Column(BigInteger, nullable=True)


class Sample(
    SampleMixin,
    StorageBase,
    metaclass=PartitionByMeta,
    partition_by="dataset_id",  # type: ignore
    partition_type="LIST",  # type: ignore
):
    """Sample model."""

    __tablename__ = "samples"
    # See https://docs.sqlalchemy.org/en/13/core/metadata.html?highlight=extend_existing#sqlalchemy.schema.Table.params.extend_existing  # noqa: E501
    __table_args__ = {"extend_existing": True}

    file = relationship("File")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Sample {self.sample_id}>"

    @staticmethod
    def add_dataset(dataset_id: int, session: Session, engine: Engine, hash_partition_modulus: int = 64) -> None:
        partition_stmt = f"FOR VALUES IN ({dataset_id})"
        partition_suffix = f"_did{dataset_id}"
        dataset_partition = Sample._create_partition(
            Sample,
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by="sample_id",
            subpartition_type="HASH",
            session=session,
            engine=engine,
        )

        if dataset_partition is None:
            return  # partitoning disabled

        # Create partitions for sample key hash
        for i in range(hash_partition_modulus):
            partition_suffix = f"_part{i}"
            partition_stmt = f"FOR VALUES WITH (modulus {hash_partition_modulus}, remainder {i})"
            _ = Sample._create_partition(
                dataset_partition,
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
        subpartition_by: Optional[str],
        subpartition_type: Optional[str],
        session: Session,
        engine: Engine,
    ) -> Optional[PartitionByMeta]:
        """Create a partition for the Sample table."""
        #  If sqlite, do not partition
        if session.bind.dialect.name == "sqlite":
            return None

        #  Create partition
        partition = instance.create_partition(
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by=subpartition_by,
            subpartition_type=subpartition_type,
            unlogged=False,
        )

        #  Create table
        Sample.metadata.create_all(engine, [partition.__table__])

        return partition
