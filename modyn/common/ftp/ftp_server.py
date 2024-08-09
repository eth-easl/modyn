"""FTP server context manager."""

import logging
import pathlib
import threading

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import ThreadedFTPServer as pyFTPServer

logger = logging.getLogger(__name__)


class FTPServer:
    """FTPServer for sending/receiving files."""

    def __init__(self, ftp_port: str, serving_directory: pathlib.Path) -> None:
        """Initialize the FTP server.

        Args:
            ftp_port: port on which ftp server is running.
            serving_directory: directory from which the ftp server is served.
        """
        self.ftp_port = ftp_port
        self.serving_directory = serving_directory
        self.authorizer = DummyAuthorizer()
        # TODO(#180): Only allow connections as soon as it has a ip?
        self.authorizer.add_user("modyn", "modyn", str(self.serving_directory), perm="elradfmwMT")

        self.handler = FTPHandler  # Intentionally a class reference
        self.handler.authorizer = self.authorizer

        self.address = ("", self.ftp_port)
        self.thread = threading.Thread(target=self.create_server_and_serve)
        self.server: pyFTPServer | None = None

    def create_server_and_serve(self) -> None:
        self.server = pyFTPServer(self.address, self.handler)
        logger.debug(f"Run FTP server on port {self.ftp_port}.")
        self.server.serve_forever()

    def __enter__(self) -> pyFTPServer:
        """Enter the context manager.

        Returns:
            pyFTPServer: the ftp server.
        """

        self.thread.start()  # As serve_forever is blocking, we run it in another thread.
        return self.server

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
        logger.debug("Closed FTP server.")
