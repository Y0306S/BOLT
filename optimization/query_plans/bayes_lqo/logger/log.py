import os
import sys

from constants import USE_LOGGER
from loguru import _Logger
from loguru import logger as logger_init

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

if USE_LOGGER:
    logger_init.remove()
    logger_init.add(
        sys.stdout,
        level=os.getenv("LOG_LEVEL") or "WARNING",
        backtrace=True,
        diagnose=True,
        format="<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    l = logger_init
else:

    class NoOpLogger(_Logger):
        def __init__(self):
            super().__init__(
                core=None,
                exception=None,
                depth=0,
                record=False,
                lazy=False,
                colors=False,
                raw=False,
                capture=True,
                patchers=[],
                extra={},
            )

        @staticmethod
        def debug(*args, **kwargs):
            pass

        @staticmethod
        def info(*args, **kwargs):
            pass

        @staticmethod
        def warning(*args, **kwargs):
            pass

        @staticmethod
        def error(*args, **kwargs):
            pass

    l = NoOpLogger()
