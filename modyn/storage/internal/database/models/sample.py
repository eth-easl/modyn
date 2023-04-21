"""Sample model."""

from typing import Any, Optional

from modyn.database import PartitionByMeta
from modyn.storage.internal.database.storage_base import StorageBase
from sqlalchemy import BigInteger, Column, Integer
from sqlalchemy.dialects import sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session
from sqlalchemy.schema import PrimaryKeyConstraint

BIGINT = BigInteger().with_variant(sqlite.INTEGER(), "sqlite")


class SampleMixin:
    # Note that we have a composite primary key in the general case because partitioning on the dataset
    # requires the dataset_id to be part of the PK.
    # Logically, sample_id is sufficient for the PK.
    sample_id = Column("sample_id", BIGINT, autoincrement=True, primary_key=True)
    dataset_id = Column(Integer, nullable=False, primary_key=True)
    file_id = Column(Integer, nullable=True)
    # This should not be null but we remove the integrity check in favor of insertion performance.
    index = Column(BigInteger, nullable=True)
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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Sample {self.sample_id}>"

    @staticmethod
    def ensure_pks_correct(session: Session) -> None:
        if session.bind.dialect.name == "sqlite":
            # sqllite does not support AUTOINCREMENT on composite PKs
            # As it also does not support partitioning, in case of sqlite, we need to update the model
            # to only have sample_id as PK, which has no further implications. dataset_id is only part of the PK
            # in the first case as that is required by postgres for partitioning.
            # Updating the model at runtime requires hacking the sqlalchemy internals
            # and what exactly do change took me a while to figure out.
            # This is not officially supported by sqlalchemy.
            # Basically, we need to change all the things where dataset_id is part of the PK
            # Simply writing Sample.dataset_id.primary_key = False or
            # Sample.dataset_id = Column(..., primary_key=False) does not work at runtime.
            # We first need to mark the column as non primary key
            # and then update the constraint (on the Table object, used to create SQL operations)
            # Last, we have to update the mapper
            # (used during query generation, needs to be synchronized to the Table, otherwise we get an error)
            if Sample.__table__.c.dataset_id.primary_key:
                Sample.__table__.c.dataset_id.primary_key = False
                Sample.__table__.primary_key = PrimaryKeyConstraint(Sample.sample_id)
                Sample.__mapper__.primary_key = Sample.__mapper__.primary_key[0:1]

    @staticmethod
    def add_dataset(dataset_id: int, session: Session, engine: Engine, hash_partition_modulus: int = 8) -> None:
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
            Sample._create_partition(
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
        if session.bind.dialect.name == "sqlite":
            return None

        # Create partition
        partition = instance.create_partition(
            partition_suffix,
            partition_stmt=partition_stmt,
            subpartition_by=subpartition_by,
            subpartition_type=subpartition_type,
            unlogged=True,  # We trade-off postgres crash resilience against performance (this gives a 2x speedup)
        )

        #  Create table
        Sample.metadata.create_all(engine, [partition.__table__])

        return partition
