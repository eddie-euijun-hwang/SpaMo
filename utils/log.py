import logging


def get_error_logger(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger(f"{__name__}.errors")
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(funcName)s - %(lineno)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
