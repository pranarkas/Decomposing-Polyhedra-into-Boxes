import logging
import sys


class ColorFormatter(logging.Formatter):
    """Colorful log formatter using ANSI escape sequences."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"

    COLORS = {
        logging.DEBUG: "\x1b[34m",  # Blue
        logging.INFO: "\x1b[32m",  # Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",  # Red
        logging.CRITICAL: "\x1b[35m",  # Magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        level_color = self.COLORS.get(record.levelno, "")
        prefix = f"{self.BOLD}{level_color}{record.levelname:8}{self.RESET}"
        fmt = f"%(asctime)s {prefix} %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logger(level: str = "INFO") -> None:
    """Configure root logger with colored output to stdout."""
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            h.setFormatter(ColorFormatter())
            h.setLevel(level.upper())
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColorFormatter())
    handler.setLevel(level.upper())
    root.addHandler(handler)
    root.setLevel(level.upper())
