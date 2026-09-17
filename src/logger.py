"""
HandsToVoice — Centralized Logging

Replaces ad-hoc print() calls with a leveled, timestamped logger shared
across the whole system: console output for interactive use, plus a
rotating file log (logs/handstovoice.log) that survives after the app
closes, for post-hoc debugging and training-run history.

Usage:
    from src.logger import get_logger
    logger = get_logger("detector")
    logger.info("Camera opened on index %d", camera_index)
    logger.warning("No hand detected in frame")
    logger.error("Failed to load model: %s", exc)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_REPO_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "handstovoice.log")

_NAMESPACE = "handstovoice"
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"

_configured = False
_console_handler = None


def _configure_root(console_level=logging.INFO):
    global _configured, _console_handler
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger(_NAMESPACE)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(console_level)
    _console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(_console_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(file_handler)

    _configured = True


def get_logger(name):
    """Return a logger under the shared 'handstovoice' namespace.

    `name` is a short module tag (e.g. "detector", "gui.main_window"),
    not the raw __name__ — keeps log lines readable instead of showing
    full dotted import paths like "src.gui.main_window".
    """
    _configure_root()
    return logging.getLogger(f"{_NAMESPACE}.{name}")


def set_console_level(level):
    """Raise/lower console verbosity at runtime (e.g. a --debug CLI flag).
    The file log always stays at DEBUG regardless of this setting.
    """
    _configure_root()
    _console_handler.setLevel(level)
