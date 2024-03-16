import logging
import sys
from fastcore.basics import store_attr
from logging.handlers import TimedRotatingFileHandler

class Logger:
    """Handles logging setup and config.
    
    NOTE: meant to be used when we are running in the cloud.
    """

    def __init__(
        self,
        log_level: str = "INFO",
        log_file: str = None,
        log_rotation: bool = True,
        log_retention: int = 60,
        log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        log_date_format: str = "%Y-%m-%d %H:%M:%S",
        do_setup: bool = True,
    ):
        """
        Initializes the LoggingManager.

        Args:
            log_level (str): The minimum log level to capture (default: "INFO").
            log_file (str): The path to the log file (default: None, logs to console only).
            log_rotation (bool): Whether to enable log rotation (default: True).
            log_retention (int): The number of days to retain log files (default: 60).

        Example:
            # import the 
            from chaski.utils.logging import Logger

            # Set up logging
            logging_manager = Logger(log_level="INFO", log_file="app.log", log_rotation=True, log_retention=60)

            # Create a logger for the current module
            logger = Logger.get_logger(__name__)
        """
        store_attr()
        if do_setup: self._setup_logging()

    def _setup_logging(self):
        """Sets up the logging configuration."""
        # Set up the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.getLevelName(self.log_level))

        # Create a console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.getLevelName(self.log_level))
        console_formatter = logging.Formatter(self.log_format, datefmt=self.log_date_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        if self.log_file:
            # Create a file handler
            if self.log_rotation:
                file_handler = TimedRotatingFileHandler(
                    self.log_file, when="midnight", backupCount=self.log_retention
                )
            else:
                file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.getLevelName(self.log_level))
            file_formatter = logging.Formatter(self.log_format, datefmt=self.log_date_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Disable logging for third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Returns a logger with the specified name.

        Args:
            name (str): The name of the logger.

        Returns:
            logging.Logger: The logger instance.
        """
        return logging.getLogger(name)