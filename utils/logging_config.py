import logging
from logging import StreamHandler


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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
