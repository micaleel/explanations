import sys

from logbook import StreamHandler, Logger

from explanations.singles import Singleton


class LoggerSingle(Logger):
    __metaclass__ = Singleton


def get_logger(format_string=None):
    """Returns a singleton instance of a LogBook Logger

    Args:
        format_string: specifies how the log messages should be formatted

    Returns:
        A logbook Logger
    """
    if format_string is None:
        format_string = (
            u'[{record.time:%Y-%m-%d %H:%M:%S.%f} pid({record.process})] ' +
            u'{record.level_name}: {record.module}::{record.func_name}:{record.lineno} {record.message}'
        )
    # default_handler = StderrHandler(format_string=log_format)
    default_handler = StreamHandler(sys.stdout, format_string=format_string)
    default_handler.push_application()
    return LoggerSingle(__name__)
