"""
Logging and metrics utilities
"""

import functools
import logging
import os
import sys
import time
import uuid

import attr
import structlog

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)

logger = structlog.get_logger()

__id__ = 'mapmatcher'


def setup_logging():
    logging.basicConfig(format="%(message)s",
                        stream=sys.stdout,
                        level=LOG_LEVEL)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(sort_keys=True)
        ],
        context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    log = logging.getLogger('werkzeug')
    log.disabled = True


def get_logger():
    return logger.new(id=__id__)


@attr.s
class observer(object):
    """
    Out-of-the box structured debug logging for a callable.

    NOTE: currently setup to work for *class* methods.
    """

    logger = attr.ib(default=get_logger())

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(self_, *args, **kwargs):
            fn_name = fn.__name__
            try:
                ret = fn(self_, *args, **kwargs)
                args_ = {'arg_%d' % i: args[i] for i in range(len(args))}
                self.logger.debug(
                    fn_name,
                    epoch=time.monotonic(),
                    return_value=ret,
                    **args_,
                    **kwargs,
                )
                return ret
            except Exception as exc:
                self.logger.debug(fn_name, exception=str(exc))
                raise exc
            return ret

        return wrapped
