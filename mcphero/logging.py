import logging
import sys
from typing import TextIO


def get_logger(name: str, log_level: int, stream: "TextIO", fmt: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    handler = logging.StreamHandler(stream=stream)
    logger.addHandler(handler)
    return logger


logger = get_logger(
    name="mcphero",
    log_level=logging.INFO,
    stream=sys.stderr,
    fmt="%(asctime)s %(levelname)8s - %(message)s",
)
