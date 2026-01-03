"""Logging configuration for baseball biomechanics system."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default logging configuration
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file (optional).
        log_format: Log message format.
        date_format: Date format for log messages.
        max_bytes: Maximum size of log file before rotation.
        backup_count: Number of backup log files to keep.
        console_output: Whether to output logs to console.

    Returns:
        Root logger instance.
    """
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set levels for noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.debug(f"Logging configured: level={level}, file={log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger property.

    Classes that inherit from this mixin will have access to a
    self.logger attribute configured for that class.

    Example:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger


def log_exception(
    logger: logging.Logger,
    message: str = "An exception occurred",
) -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance to use.
        message: Message to log along with the exception.
    """
    logger.exception(message)


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.

    Logs progress at configurable intervals to avoid flooding the log.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        description: str = "Processing",
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance to use.
            total: Total number of items to process.
            description: Description of the operation.
            log_interval: Log every N percent.
        """
        self.logger = logger
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = -log_interval

    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.

        Args:
            n: Number of items processed.
        """
        self.current += n
        percent = int((self.current / self.total) * 100)

        if percent >= self.last_logged_percent + self.log_interval:
            self.logger.info(
                f"{self.description}: {percent}% ({self.current}/{self.total})"
            )
            self.last_logged_percent = percent

    def finish(self) -> None:
        """Log completion of the operation."""
        self.logger.info(
            f"{self.description}: Complete ({self.current}/{self.total})"
        )
