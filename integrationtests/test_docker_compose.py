# After running docker-compose up, this script checks whether all services are reachable within a reasonable timeout
import os
import pathlib
import time

import grpc
import psycopg2
import yaml
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub  # noqa: F401
from modyn.utils import grpc_connection_established

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 60  # seconds
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"


def terminate_on_timeout(start_time: int) -> None:
    curr_time = round(time.time())

    if curr_time - start_time < TIMEOUT:
        return

    raise TimeoutError("Reached timeout")


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def storage_running() -> bool:
    config = get_modyn_config()

    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    storage_channel = grpc.insecure_channel(storage_address)

    if not grpc_connection_established(storage_channel):
        print(f"Could not establish gRPC connection to storage at {storage_address}. Retrying.")
        return False

    return True


def storage_db_running() -> bool:
    config = get_modyn_config()
    try:
        psycopg2.connect(
                        host=config['storage']['database']['host'],
                        port=config['storage']['database']['port'],
                        database=config['storage']['database']['database'],
                        user=config['storage']['database']['username'],
                        password=config['storage']['database']['password'],
                        connect_timeout=5
                    )

        return True
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to the database: " + str(error))
        return False


def metadata_db_running() -> bool:
    config = get_modyn_config()
    try:
        psycopg2.connect(
                        host=config['metadata_database']['host'],
                        port=config['metadata_database']['port'],
                        database=config['metadata_database']['database'],
                        user=config['metadata_database']['username'],
                        password=config['metadata_database']['password'],
                        connect_timeout=5
                    )

        return True
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to the database: " + str(error))
        return False


def selector_running() -> bool:
    # TODO(MaxiBoether): implement this when selector is merged and docker entrypoint works
    return True


def system_running() -> bool:
    return storage_db_running() and storage_running() and selector_running() and metadata_db_running()


def main() -> None:
    start_time = round(time.time())

    while True:
        if system_running():
            return

        terminate_on_timeout(start_time)
        time.sleep(1)


if __name__ == '__main__':
    main()
