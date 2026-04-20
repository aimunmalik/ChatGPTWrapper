import json
import logging
import sys
from typing import Any

PHI_FIELDS = frozenset(
    {
        "message",
        "content",
        "messages",
        "assistantMessage",
        "userMessage",
        "text",
        "body",
        "prompt",
        "completion",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            # Never emit the traceback text: library exceptions (e.g. openpyxl
            # on a bad cell, python-docx on a malformed paragraph) often embed
            # user-supplied content in their message, and the traceback frames
            # can surface local variables that hold PHI. Record only the
            # exception class name so ops can see WHAT failed, not WHY.
            exc_type = record.exc_info[0]
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else "Exception"
            }

        for key, value in record.__dict__.items():
            if key in PHI_FIELDS:
                continue
            if key.startswith("_"):
                continue
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "message",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            }:
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
