import logging
import os
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

_LOG_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def configure_logging(level=logging.INFO):
    """Configure root logging for the application.

    level may be an int or a string name like 'INFO' or 'DEBUG'. If a string is
    provided we map it to the corresponding numeric level. If handlers already
    exist on the root logger we leave configuration alone to avoid duplicate
    handlers when invoked multiple times.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = StreamHandler()
    handler.setFormatter(_LOG_FORMATTER)
    root.addHandler(handler)
    root.setLevel(level)


def add_file_handler(log_path: str, level=logging.DEBUG) -> None:
    """Attach a FileHandler to the root logger so all log records are written
    to *log_path* in addition to the existing stream handlers.

    The parent directory is created automatically if it does not exist.
    Calling this function more than once with the same path is safe — it will
    not add duplicate handlers.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.DEBUG)

    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    # Avoid duplicate file handlers pointing to the same file
    for h in root.handlers:
        if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
            if os.path.abspath(getattr(h, 'baseFilename', '')) == log_path:
                return

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(_LOG_FORMATTER)
    file_handler.setLevel(level)
    root.addHandler(file_handler)
    # Ensure root level is at most the handler level so records reach the handler
    if root.level > level:
        root.setLevel(level)
