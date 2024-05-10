import logging


def setup_logging(file: str, level: int = logging.NOTSET) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    return logging.getLogger(file)
