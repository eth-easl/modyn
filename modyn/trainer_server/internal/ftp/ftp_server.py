"""FTP server context manager."""

import logging
import pathlib
import threading
from typing import Optional

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import ThreadedFTPServer as pyFTPServer

logger = logging.getLogger(__name__)


class FTPServer:
    """FTPServer for sending/receiving models."""

    def __init__(self, config: dict, temp_directory: pathlib.Path) -> None:
        """Initialize the FTP server.

        Args:
            config (dict): Modyn configuration.
        """
        self.config = config
        self.serving_directory = temp_directory
        self.authorizer = DummyAuthorizer()
        # TODO(create issue): Only allow connections from supervisor as soon as it has a ip?
        self.authorizer.add_user("modyn", "modyn", str(self.serving_directory), perm="elradfmwMT")

        self.handler = FTPHandler  # Intentionally a class reference
        self.handler.authorizer = self.authorizer

        self.address = ("", self.config["trainer_server"]["ftp_port"])
        self.thread = threading.Thread(target=self.create_server_and_serve)
        self.server: Optional[pyFTPServer] = None

    def create_server_and_serve(self) -> None:
        self.server = pyFTPServer(self.address, self.handler)
        self.server.serve_forever()

    def __enter__(self) -> pyFTPServer:
        """Enter the context manager.

        Returns:
            FTPServer: self
        """

        self.thread.start()  # As serve_forever is blocking, we run it in another thread.
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        assert self.server is not None
        self.server.close_all()  # Blocks until server stopped.
        self.thread.join()  # Wait for thread cleanup.
        logger.debug("Closed FTP Server.")
