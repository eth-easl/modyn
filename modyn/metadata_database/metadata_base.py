"""Base class for metadata database."""

import logging
from typing import Any, Optional

from sqlalchemy import event
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.decl_api import DeclarativeAttributeIntercept
from sqlalchemy.sql.ddl import DDL

logger = logging.getLogger(__name__)
# Kudos: https://stackoverflow.com/questions/61545680/postgresql-partition-and-sqlalchemy


class PartitionByMeta(DeclarativeAttributeIntercept):
    def __new__(
        cls,
        clsname: str,
        bases: Any,
        attrs: dict[str, Any],
        *,
        partition_by: Optional[str],
        partition_type: Optional[str],
    ) -> DeclarativeAttributeIntercept:
        @classmethod  # type: ignore # (see https://github.com/python/mypy/issues/4153)
        def get_partition_name(cls_: PartitionByMeta, suffix: str) -> str:
            return f"{cls_.__tablename__}_{suffix}"

        @classmethod  # type: ignore # (see https://github.com/python/mypy/issues/4153)
        def create_partition(
            cls_: PartitionByMeta,
            suffix: str,
            partition_stmt: str,
            subpartition_by: Optional[str] = None,
            subpartition_type: Optional[str] = None,
            unlogged: bool = True,
        ) -> PartitionByMeta:
            if suffix not in cls_.partitions:
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

        if partition_by is not None and partition_type is not None:
            table_args = attrs.get("__table_args__", ())
            if isinstance(table_args, dict):
                table_args["postgresql_partition_by"] = f"{partition_type.upper()}({partition_by})"
            else:
                table_args += ({"postgresql_partition_by": f"{partition_type.upper()}({partition_by})"},)
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
