import logging
import logging.handlers

LOGGING_LEVEL = logging.INFO

stdlogger = logging.getLogger('icl')
stdlogger.setLevel(level=LOGGING_LEVEL)
