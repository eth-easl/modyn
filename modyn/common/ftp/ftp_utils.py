# Utils file containing functions in order to simplify FTP server interactions.
import pathlib
from ftplib import FTP
from typing import Any, Callable, Optional


def download_file(
    hostname: str,
    port: int,
    user: str,
    password: str,
    remote_file_path: pathlib.Path,
    local_file_path: pathlib.Path,
    callback: Optional[Callable[[float], None]] = None,
) -> None:
    """Downloads a file from a given host to the local filesystem. If the file already exists, it gets overwritten.

    Args:
        hostname: the host from which to download the file.
        port: ftp port of the host.
        user: username used to login.
        password: the password of the user.
        remote_file_path: path to the remote file.
        local_file_path: local path to the file.
        callback(int, int): function called every block of data with the total size and the block size.

    Returns:

    """
    ftp = FTP()
    ftp.connect(hostname, port, timeout=3)

    ftp.login(user, password)
    ftp.sendcmd("TYPE i")  # Switch to binary mode
    size = ftp.size(str(remote_file_path))

    assert size, f"Could not read size of file with path {remote_file_path} from FTP server."

    with open(local_file_path, "wb") as local_file:

        def write_callback(data: Any) -> None:
            local_file.write(data)
            if callback:
                nonlocal size
                callback(float(len(data)) / size)  # type: ignore

        ftp.retrbinary(f"RETR {remote_file_path}", write_callback)

    ftp.close()


def upload_file(
    hostname: str, port: int, user: str, password: str, local_file_path: pathlib.Path, remote_file_path: pathlib.Path
) -> None:
    """Uploads a file from the local filesystem to a given ftp server.

    Args:
        hostname: the host from which to download the file.
        port: ftp port of the host.
        user: username used to login.
        password: the password of the user.
        local_file_path: local path to the file.
        remote_file_path: path on the remote ftp server to which the file gets written.

    Returns:

    """
    ftp = FTP()
    ftp.connect(hostname, port, timeout=3)
    ftp.login(user, password)
    ftp.sendcmd("TYPE i")  # Switch to binary mode

    with open(local_file_path, "rb") as local_file:
        ftp.storbinary(f"STOR {remote_file_path}", local_file)

    ftp.close()


def delete_file(hostname: str, port: int, user: str, password: str, remote_file_path: pathlib.Path) -> None:
    """Delete a file from a remote ftp server.

    Args:
        hostname: the host from which to download the file.
        port: ftp port of the host.
        user: username used to login.
        password: the password of the user.
        remote_file_path: path to the file on the remote ftp server which should be deleted.

    Returns:

    """
    ftp = FTP()
    ftp.connect(hostname, port, timeout=3)
    ftp.login(user, password)
    ftp.delete(str(remote_file_path))
    ftp.close()
