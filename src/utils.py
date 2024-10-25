import logging
import os


def setup_logging(log_level="INFO", log_file=None, console_log=True):
    """Setup logging configuration.

    Parameters
    ----------
    log_level : str
        The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        The file path to write logs (default is None).
    console_log : bool, optional
        Whether to log to console (default is True).
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    handlers = []
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    if console_log:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), "INFO"),
        format=log_format,
        handlers=handlers,
    )
