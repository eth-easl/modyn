# pylint: disable=too-many-instance-attributes
from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session


class AbstractDatabaseConnection(ABC):
    def __init__(self, modyn_config: dict, print_queries: bool = False) -> None:
        """Initialize the database connection.

        Args:
            modyn_config (dict): Configuration of the modyn module.
            print_queries (bool): Whether to log the queries to DB that SQLAlchemy makes
        """
        self.modyn_config = modyn_config
        self.print_queries = print_queries
        self.session: Session = None
        self.engine: Engine = None
        self.url = None
        self.drivername: str | None = None
        self.username: str | None = None
        self.password: str | None = None
        self.host: str | None = None
        self.port: int | None = None
        self.database: str | None = None

    def setup_connection(self) -> None:
        self.url = URL.create(
            drivername=self.drivername,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        self.engine = create_engine(self.url, echo=self.print_queries)
        self.session = sessionmaker(bind=self.engine)()

    def terminate_connection(self) -> None:
        self.session.close()
        self.engine.dispose()

    def __enter__(self) -> AbstractDatabaseConnection:
        """Create the engine and session.

        Returns:
            DatabaseConnection: DatabaseConnection.
        """
        self.setup_connection()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Close the session and dispose the engine.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        self.terminate_connection()

    @abstractmethod
    def create_tables(self) -> None:
        """Create all tables."""
