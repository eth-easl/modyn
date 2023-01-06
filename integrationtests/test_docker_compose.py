# After running docker-compose up, this script checks whether all services are reachable within a reasonable timeout
import time
import os
import pathlib
import yaml
import psycopg2

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))

TIMEOUT = 10  # seconds
CONFIG_FILE = SCRIPT_PATH.parent.parent / "modyn" / "config" / "config.yaml"


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
    return True


def storage_db_running() -> bool:
    # config = get_modyn_config()
    # conn = None
    try:
        # psycopg2.connect(
        #                host=config['odm']['postgresql']['host'],
        #                port=config['odm']['postgresql']['port'],
        #                database=config['odm']['postgresql']['database'],
        #                user=config['odm']['postgresql']['user'],
        #                password=config['odm']['postgresql']['password'],
        #                connect_timeout=5
        #            )

        # TODO(Maxiboether): replace with storage DB after merge of storage PR

        return True
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return False


def selector_running() -> bool:
    # TODO(MaxiBoether): implement this when storage is merged and docker entrypoint works
    return True


def system_running() -> bool:
    return storage_db_running() and storage_running() and selector_running()


def main() -> None:
    start_time = round(time.time())

    while True:
        if system_running():
            return

        terminate_on_timeout(start_time)
        time.sleep(1)


if __name__ == '__main__':
    main()
