import time
from ftplib import FTP

from integrationtests.utils import get_modyn_config

TIMEOUT = 60  # in seconds


def ftp_running() -> bool:
    return model_storage_ftp_running() and trainer_server_ftp_running()


def model_storage_ftp_running() -> bool:
    config = get_modyn_config()

    ftp = FTP()
    try:
        ftp.connect(config["model_storage"]["hostname"], int(config["model_storage"]["ftp_port"]), timeout=3)
        ftp.login("modyn", "modyn")
    except (ConnectionRefusedError, TimeoutError):
        return False
    finally:
        ftp.close()

    return True


def trainer_server_ftp_running() -> bool:
    config = get_modyn_config()

    ftp = FTP()
    try:
        ftp.connect(config["trainer_server"]["hostname"], int(config["trainer_server"]["ftp_port"]), timeout=3)
        ftp.login("modyn", "modyn")
    except (ConnectionRefusedError, TimeoutError):
        return False
    finally:
        ftp.close()

    return True


def terminate_on_timeout(start_time: int) -> None:
    curr_time = round(time.time())

    if curr_time - start_time < TIMEOUT:
        return

    raise TimeoutError("Reached timeout")


def main() -> None:
    start_time = round(time.time())

    while True:
        if ftp_running():
            return

        terminate_on_timeout(start_time)
        time.sleep(1)


if __name__ == "__main__":
    main()
