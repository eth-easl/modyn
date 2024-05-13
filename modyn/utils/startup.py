import logging
import multiprocessing as mp
import os


def set_start_method_spawn(logger: logging.Logger) -> None:
    # At the top because the FTP Server or other dependencies otherwise set fork.
    try:
        if mp.get_start_method() != "spawn":
            mp.set_start_method("spawn")
    except RuntimeError as error:
        if mp.get_start_method() != "spawn" and "PYTEST_CURRENT_TEST" not in os.environ:
            logger.error("Start method is already set to {}", mp.get_start_method())
        raise error
