import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(
        log_level: str = "INFO", 
        log_file: str = None, 
        log_rotation: bool = True,
        log_retention: int = 60,
    ):
    """Sets up logging for the application.

    Args:
        log_level (str): The minimum log level to capture (default: "INFO").
        log_file (str): The path to the log file (default: None, logs to console only).
        log_rotation (bool): Whether to enable log rotation (default: True).
        log_retention (int): The number of days to retain log files (default: 60).

    Examples:
        from chaski.utils.logging_utils import setup_logging
        setup_logging(log_level="INFO", log_file="app.log", log_rotation=True, log_retention=60)
    """
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"

    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(log_level))

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.getLevelName(log_level))
    console_formatter = logging.Formatter(log_format, datefmt=log_date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        # Create a file handler
        if log_rotation:
            file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=log_retention)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.getLevelName(log_level))
        file_formatter = logging.Formatter(log_format, datefmt=log_date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Disable logging for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)