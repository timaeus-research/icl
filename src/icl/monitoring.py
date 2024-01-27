import logging
import logging.handlers

LOG_FILE_NAME = 'icl.log'
LOGGING_LEVEL = logging.INFO

# formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a', maxBytes=5000000, backupCount=5)
# handler.setFormatter(formatter)
stdlogger = logging.getLogger('icl')
stdlogger.addHandler(handler)
stdlogger.setLevel(level=LOGGING_LEVEL)
