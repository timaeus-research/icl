import logging

LOG_FILE_NAME = '/var/log/icl.log'
LOGGING_LEVEL = logging.INFO

handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a', maxBytes=5000000, backupCount=5)

stdlogger = logging.getLogger('icl')
stdlogger.addHandler(handler)
stdlogger.setLevel(level=LOGGING_LEVEL)
