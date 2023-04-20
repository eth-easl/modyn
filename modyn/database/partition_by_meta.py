"""Base class for metadata database."""

import logging
from typing import Any, Optional

from sqlalchemy import event
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
        *,  # This is to make this implementation of more compatible with the original, we don't use *args because pylint complains otherwise  # noqa: E501
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
            additional_table_args: Optional[tuple[Any]] = None,
        ) -> PartitionByMeta:
            if suffix not in cls_.partitions:
                #  attrs are the attributes of the class that is being created
                #  We need to update the __tablename__ attribute to the name of the partition
                attrs = {"__tablename__": cls_.get_partition_name(suffix)}

                if additional_table_args is not None and len(additional_table_args) > 0:
                    attrs["__table_args__"] = additional_table_args

                    if unlogged:
                        attrs["__table_args__"] += ({"prefixes": ["UNLOGGED"]},)
                else:
                    if unlogged:
                        attrs["__table_args__"] = {"prefixes": ["UNLOGGED"]}

                #  We then pass the updated attributes to the PartitionByMeta class
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
            # The attrs are the attributes of the class that is being created
            # We need to update the __table_args__ attribute to include the partitioning information
            if "__table_args__" not in attrs:
                attrs["__table_args__"] = ()
            table_args = attrs["__table_args__"]
            # We need the following check because __table_args__ can be a tuple or a dict
            # See here for more details:
            # https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html#orm-declarative-table-configuration
            if isinstance(table_args, dict):
                table_args["postgresql_partition_by"] = f"{partition_type.upper()}({partition_by})"
            else:
                if len(table_args) == 0 or table_args[-1] is None or not isinstance(table_args[-1], dict):
                    table_args += ({"postgresql_partition_by": f"{partition_type.upper()}({partition_by})"},)
                else:
                    table_args[-1]["postgresql_partition_by"] = f"{partition_type.upper()}({partition_by})"

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
