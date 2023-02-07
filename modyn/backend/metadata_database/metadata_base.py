"""Base class for metadata database backends."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
