"""Base class for metadata database."""

from sqlalchemy import event
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.ddl import DDL
from sqlalchemy.orm.decl_api import DeclarativeAttributeIntercept
import traceback

# Kudos: https://stackoverflow.com/questions/61545680/postgresql-partition-and-sqlalchemy


class PartitionByMeta(DeclarativeAttributeIntercept):
    def __new__(cls, clsname, bases, attrs, *, partition_by, partition_type):
        @classmethod
        def get_partition_name(cls_, suffix):
            return f"{cls_.__tablename__}_{suffix}"

        @classmethod
        def create_partition(cls_, suffix, partition_stmt, subpartition_by=None, subpartition_type=None, unlogged=True):
            print(f"Creating partition {suffix} for {cls_.__tablename__} with statement {partition_stmt}")
            print(f"Currently known partitions: {cls_.partitions}")
            if suffix not in cls_.partitions:  # TODO: what happens on restart? how do we handle existing partitions?
                attrs = {"__tablename__": cls_.get_partition_name(suffix)}
                if unlogged:
                    attrs["__table_args__"] = {"prefixes": ["UNLOGGED"]}

                partition = PartitionByMeta(
                    f"{clsname}{suffix}",
                    bases,
                    attrs,
                    partition_type=subpartition_type,
                    partition_by=subpartition_by,
                )

                partition.__table__.add_is_dependent_on(cls_.__table__)

                event.listen(
                    partition.__table__,
                    "after_create",
                    DDL(
                        f"""
                        ALTER TABLE {cls_.__tablename__}
                        ATTACH PARTITION {partition.__tablename__}
                        {partition_stmt};
                        """
                    ),
                )

                cls_.partitions[suffix] = partition

            return cls_.partitions[suffix]
    
        if partition_by is not None:
            table_args = attrs.get("__table_args__", ())
            if type(table_args) is dict:
                table_args["postgresql_partition_by"] = f'{partition_type.upper()}({partition_by})'
            else:
                table_args += ({"postgresql_partition_by": f'{partition_type.upper()}({partition_by})'},)
            attrs.update(
                {
                    "__table_args__": table_args,
                    "partitions": {},
                    "partitioned_by": partition_by,
                    "get_partition_name": get_partition_name,
                    "create_partition": create_partition,
                }
            )

        return super().__new__(cls, clsname, bases, attrs)


class MetadataBase(DeclarativeBase):
    pass
