# Utils file containing functions in order to simplify FTP server interactions.
import logging
import pathlib
from ftplib import FTP
from logging import Logger
from typing import Any, Callable, Optional

from modyn.utils import EMIT_MESSAGE_PERCENTAGES, calculate_checksum


def download_file(
    hostname: str,
    port: int,
    user: str,
    password: str,
    remote_file_path: pathlib.Path,
    local_file_path: pathlib.Path,
    callback: Optional[Callable[[float], None]] = None,
    checksum: Optional[bytes] = None,
) -> bool:
    """Downloads a file from a given host to the local filesystem. If the file already exists, it gets overwritten.

    Args:
        hostname: the host from which to download the file.
        port: ftp port of the host.
        user: username used to login.
        password: the password of the user.
        remote_file_path: path to the remote file.
        local_file_path: local path to the file.
        callback(float): function called every block of data with the current progress in [0, 1].
        checksum: the expected hash of the file.
    Returns:
        bool: whether the file was successfully downloaded.
    """
    ftp = FTP()
    ftp.connect(hostname, port, timeout=5 * 60)

    ftp.login(user, password)
    ftp.sendcmd("TYPE i")  # Switch to binary mode
    size = ftp.size(str(remote_file_path))

    assert size, f"Could not read size of file with path {remote_file_path} from FTP server."

    total_downloaded = 0

    with open(local_file_path, "wb") as local_file:

        def write_callback(data: Any) -> None:
            local_file.write(data)
            if callback:
                nonlocal size, total_downloaded
                total_downloaded += len(data)
                callback(float(total_downloaded) / size)  # type: ignore

        ftp.retrbinary(f"RETR {remote_file_path}", write_callback)

    ftp.close()

    if checksum:
        local_hash = calculate_checksum(local_file_path)
        return local_hash == checksum
    return True


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
    ftp.connect(hostname, port, timeout=5 * 60)
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
    ftp.connect(hostname, port, timeout=5 * 60)
    ftp.login(user, password)
    ftp.delete(str(remote_file_path))
    ftp.close()


def get_pretrained_model_callback(logger: Logger) -> Callable[[float], None]:
    """Creates the standard callback used to download a pretrained model.

    Args:
        logger: to log the events.

    Returns:
        Callable[[float], None]: the callback function.
    """
    last_progress = 0.0

    def download_callback(current_progress: float) -> None:
        nonlocal last_progress
        for emit_perc in EMIT_MESSAGE_PERCENTAGES:
            if last_progress <= emit_perc < current_progress:
                logger.info(f"Completed {emit_perc * 100}% of the pretrained model download.")
        last_progress = current_progress

    return download_callback


def download_trained_model(
    logger: logging.Logger,
    model_storage_config: dict,
    remote_path: pathlib.Path,
    checksum: bytes,
    identifier: int,
    base_directory: pathlib.Path,
    pipeline_id: Optional[int] = None,
) -> Optional[pathlib.Path]:
    model_path = base_directory / f"trained_model_{identifier}.modyn"

    tries = 3

    for num_try in range(tries):
        try:
            success = download_file(
                hostname=model_storage_config["hostname"],
                port=int(model_storage_config["ftp_port"]),
                user="modyn",
                password="modyn",
                remote_file_path=remote_path,
                local_file_path=model_path,
                callback=get_pretrained_model_callback(logger),
                checksum=checksum,
            )

            if not success and num_try < tries - 1:
                logger.error("Download finished without exception but checksums did not match, retrying")
                continue
        # Retry mechanism requires generic exception
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.error("Caught exception while downloading file.")
            logger.error(ex)
            if num_try < tries - 1:
                logger.warning("Trying again")
                continue

            logger.error("Tried enough times.")
            raise

        break

    if not success:
        logger.error("Checksums did not match, evaluation cannot be started.")
        return None
    identifier_prefix = f"Pipeline ID: {pipeline_id}; Training ID: {identifier}" if pipeline_id is not None else ""
    logger.info(f"[XZM]: {identifier_prefix} Successfully downloaded the model; now delete it from the FTP server.")
    for num_try in range(tries):
        try:
            delete_file(
                hostname=model_storage_config["hostname"],
                port=int(model_storage_config["ftp_port"]),
                user="modyn",
                password="modyn",
                remote_file_path=remote_path,
            )
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.error("Caught exception while deleting file.")
            logger.error(ex)
            if num_try < tries - 1:
                logger.warning("Trying again")
                continue
            logger.error(f"[XZM]: {identifier_prefix} Tried enough times to delete the file. Give up deleting it.")
        break

    logger.info(f"Successfully downloaded trained model to {model_path}.")

    return model_path
