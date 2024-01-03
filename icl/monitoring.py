import logging

from icl.constants import LOG_FILE_NAME, LOGGING_LEVEL

handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a', maxBytes=5000000, backupCount=5)

stdlogger = logging.getLogger('icl')
stdlogger.addHandler(handler)
stdlogger.setLevel(level=LOGGING_LEVEL)
