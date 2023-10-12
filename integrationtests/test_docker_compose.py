# After running docker-compose up, this script checks whether all services are reachable within a reasonable timeout
import time

import grpc
import psycopg2
from integrationtests.utils import get_modyn_config
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub  # noqa: F401
from modyn.utils import grpc_connection_established

TIMEOUT = 180  # seconds


def terminate_on_timeout(start_time: int) -> None:
    curr_time = round(time.time())

    if curr_time - start_time < TIMEOUT:
        return

    raise TimeoutError("Reached timeout")


def storage_running() -> bool:
    config = get_modyn_config()

    storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
    storage_channel = grpc.insecure_channel(storage_address)

    if not grpc_connection_established(storage_channel):
        print(f"Could not establish gRPC connection to storage at {storage_address}. Retrying.")
        return False

    print("Sucessfully connected to storage!")

    return True


def model_storage_running() -> bool:
    config = get_modyn_config()

    model_storage_address = f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
    model_storage_channel = grpc.insecure_channel(model_storage_address)

    if not grpc_connection_established(model_storage_channel):
        print(f"Could not establish gRPC connection to model storage at {model_storage_address}. Retrying.")
        return False

    print("Sucessfully connected to model storage!")

    return True


def evaluator_running() -> bool:
    config = get_modyn_config()

    evaluator_address = f"{config['evaluator']['hostname']}:{config['evaluator']['port']}"
    evaluator_channel = grpc.insecure_channel(evaluator_address)

    if not grpc_connection_established(evaluator_channel):
        print(f"Could not establish gRPC connection to evaluator at {evaluator_address}. Retrying.")
        return False

    print("Sucessfully connected to evaluator!")

    return True


def trainer_server_running() -> bool:
    config = get_modyn_config()

    trainer_server_address = f"{config['trainer_server']['hostname']}:{config['trainer_server']['port']}"
    trainer_server_channel = grpc.insecure_channel(trainer_server_address)

    if not grpc_connection_established(trainer_server_channel):
        print(f"Could not establish gRPC connection to trainer server at {trainer_server_address}. Retrying.")
        return False

    print("Sucessfully connected to trainer server!")

    return True


def storage_db_running() -> bool:
    config = get_modyn_config()
    try:
        psycopg2.connect(
            host=config["storage"]["database"]["host"],
            port=config["storage"]["database"]["port"],
            database=config["storage"]["database"]["database"],
            user=config["storage"]["database"]["username"],
            password=config["storage"]["database"]["password"],
            connect_timeout=5,
        )

        print("Sucessfully connected to storage database!")

        return True
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to the database: " + str(error))
        return False


def metadata_db_running() -> bool:
    config = get_modyn_config()
    try:
        psycopg2.connect(
            host=config["metadata_database"]["host"],
            port=config["metadata_database"]["port"],
            database=config["metadata_database"]["database"],
            user=config["metadata_database"]["username"],
            password=config["metadata_database"]["password"],
            connect_timeout=5,
        )

        print("Sucessfully connected to metadata database!")

        return True
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to the database: " + str(error))
        return False


def selector_running() -> bool:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        print(f"Could not establish gRPC connection to selector at {selector_address}. Retrying.")
        return False

    print("Sucessfully connected to selector!")

    return True


def system_running() -> bool:
    return (
        storage_db_running()
        and storage_running()
        and selector_running()
        and metadata_db_running()
        and model_storage_running()
        and evaluator_running()
        and trainer_server_running()
    )


def main() -> None:
    start_time = round(time.time())

    while True:
        if system_running():
            return

        terminate_on_timeout(start_time)
        time.sleep(1)


if __name__ == "__main__":
    main()
