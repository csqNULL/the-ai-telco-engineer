# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-safe printing with role-based headers.

Ensures that output from parallel processes does not interleave.
Each process sets its own header (e.g. "MANAGER", "AGENT-gen00-0001").
A shared multiprocessing lock serializes writes to stdout.

Usage:
    # In the main process:
    import multiprocessing as mp
    import printer

    lock = mp.Lock()
    printer.init(lock, "MANAGER")
    printer.log("Starting optimization")

    # In a worker process:
    printer.init(lock, "WORKER-0")
    printer.log("Processing task")
    printer.set_header("AGENT-gen00-0001")
    printer.log("Rate limit hit")
"""

# TODO: should this whole module just be replaced by `logging`?

import logging
import multiprocessing as mp
import os
import re
from typing import Optional

from config import LogConfig

_lock: Optional[mp.Lock] = None
_header: str = ""
# Default logger, until it is further configured in init()
_logger: logging.Logger = logging.getLogger(__name__)
_color_map: dict[str, str] = {
    "MANAGER": "\033[32m",
    "AGENT": "\033[34m",
    "WORKER": "\033[33m",
    "EVALUATOR": "\033[35m",
    "UTILS": "\033[36m",
    "CONFIG": "\033[37m",
}
DEFAULT_COLOR = "\033[32m"


class AlignedFormatter(logging.Formatter):
    def __init__(self, formatter: logging.Formatter):
        super().__init__()
        assert not self.is_aligned_formatter(formatter)
        self.formatter = formatter

    def format(self, record):
        if isinstance(self.formatter, AlignedFormatter):
            # Not sure why, but this can happen for some reason.
            return self.formatter.format(record)

        original = self.formatter.format(record)

        first_line, *rest = original.split(os.linesep)
        if not rest:
            return first_line

        # The prefix is everything before the actual message on the first line,
        # excluding invisible characters (e.g. ANSI escape codes).
        first_line_stripped = re.sub(r'\x1b\[[0-9;]*[mK]', '', first_line)
        user_content_0 = record.message.split(os.linesep)[0]
        if not user_content_0:
            prefix_len = len(first_line_stripped)
        else:
            try:
                prefix_len = first_line_stripped.index(user_content_0) \
                             if record.message else 0
            except ValueError:
                prefix_len = 0
        padding = " " * prefix_len
        return os.linesep.join([first_line] + [padding + line for line in rest])

    @staticmethod
    def is_aligned_formatter(formatter: logging.Formatter) -> bool:
        """We can't use isinstance(formatter, AlignedFormatter) because
        different processes may load this class differently."""
        return hasattr(formatter, "is_aligned_formatter")


def init(cfg: LogConfig, lock: mp.Lock, header: str = "") -> None:
    """Initialize the printer in the current process.

    Must be called once per process. In workers, call this with the shared
    lock received from the manager and the initial header for the process.

    Args:
        lock: A multiprocessing.Lock shared across all processes.
        header: Initial header string (e.g. "MANAGER", "WORKER-0").
    """
    global _lock

    _lock = lock
    logging.basicConfig(
        level=cfg.logging_level,
        format=cfg.logging_format,
    )
    set_header(header)


def set_header(header: str) -> None:
    """Update the header for all subsequent prints in this process.

    Args:
        header: New header string (e.g. "AGENT-gen00-0003").
    """
    global _header, _logger

    _header = header

    color = _color_map.get(_header, DEFAULT_COLOR)
    prefix = f"[{color}{_header}\033[0m]" if _header else ""
    _logger = logging.getLogger(prefix)

    # Set our aligned formatter on the root logger and this one
    for h in (logging.getLogger().handlers + _logger.handlers):
        if AlignedFormatter.is_aligned_formatter(h.formatter):
            continue
        h.setFormatter(AlignedFormatter(h.formatter))


def log(*args, sep: str = " ", level: int = logging.INFO) -> None:
    """Print a single message atomically with the current header prefix.

    Safe to call from any process. Falls back to plain stdout
    if init() has not been called (e.g. during early startup).

    Args:
        *args: Values to print (joined by *sep*).
        sep: Separator between values (default: space).
    """
    message = sep.join(str(a) for a in args)
    if _lock is not None:
        with _lock:
            _logger.log(level, message)
    else:
        _logger.log(level, message)


def section(*lines: str, level: int = logging.INFO) -> None:
    """Print multiple lines as a single atomic block.

    Each non-empty line is prefixed with the header.
    Empty strings produce blank lines (no prefix).

    Args:
        *lines: One string per output line.
    """
    message = os.linesep.join(lines)
    if _lock is not None:
        with _lock:
            _logger.log(level, message)
    else:
        _logger.log(level, message)

def debug(*args, **kwargs) -> None:
    log(*args, level=logging.DEBUG, **kwargs)
def info(*args, **kwargs) -> None:
    log(*args, level=logging.INFO, **kwargs)
def warning(*args, **kwargs) -> None:
    log(*args, level=logging.WARNING, **kwargs)
def error(*args, **kwargs) -> None:
    log(*args, level=logging.ERROR, **kwargs)
def critical(*args, **kwargs) -> None:
    log(*args, level=logging.CRITICAL, **kwargs)
