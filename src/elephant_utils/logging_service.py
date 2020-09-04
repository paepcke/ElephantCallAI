'''
Created on Jan 15, 2020

@author: paepcke

Singleton-instance class to share logging among
modules. Usage:

from logging_service import LoggingService

Constructor:
            ...
        self.log = LoggingService(logfile=logfile)
        self.log.info("Constructing output file names...")
        self.log.err("Constructing output file names...")
        self.log.warn("Constructing output file names...")
        self.log.debug("Constructing output file names...")

Easily specify rotating logs. See __init__() for all option.

'''
import logging
from logging.handlers import RotatingFileHandler
import os
import sys


# ----------------------------- Metaclass ---------------------
class MetaLoggingSingleton(type):
    
    def __init__(cls, name, bases, dic):
        super(MetaLoggingSingleton, cls).__init__(name, bases, dic)
        cls.instance = None
    
    def __call__(cls, *args, **kwargs):
        if cls.instance is not None:
            return cls.instance
        cls.instance = super(MetaLoggingSingleton, cls).__call__(*args, **kwargs)
        return cls.instance

# ----------------------------- LoggingService Class ---------------------

class LoggingService(metaclass=MetaLoggingSingleton):
    '''
    A singleton logger. Use that single instance
    for multiple modules to log to one place. Methods
    are debug(), warn(), info(), and err(). Can
    log to display or to file.
    
    Easiest use is to create an instance of this class
    (which will always return the same instance for everyone).
    Assign it to a class variable called "log". Then for logging,
    use:
    
         self.log.err()
         self.log.info()
            etc.
     
    '''
        
    #-------------------------
    # __repr__ 
    #--------------
    
    def __repr__(self):
        return f'<Review judgements logging service {hex(id(self))}>'

    #-------------------------
    # Constructor 
    #--------------


    def __init__(self, 
                 logging_level=logging.INFO, 
                 logfile=None,
                 msg_identifier=None,
                 rotating_logs=True,
                 log_size=1000000,
                 max_num_logs=500,
                 logger_name=None):

        '''
        Create a shared logging service.
        Options are logging to screen, or to a file.
        Within file logging choices are whether to log
        to an ever increasing file, or to use the Python
        RotatingFileHandler facility.
        
        @param logging_level: INFO, WARN, ERR, etc as per 
            standard logging module
        @type logging_level: int
        @param logfile: if provided, file path for the log file(s)
        @type logfile: str,
        @param msg_identifier: if provided this string will
            be shown at the start of each log message. If None,
            the Python module in argv[0] will be shown without
            the .py extension.
        @type msg_identifier: {None|str}
        @param rotating_logs: whether or not to rotate logs. 
        @type rotating_logs: bool
        @param log_size: max size of each log file, if rotating 
        @type log_size: int
        @param max_num_logs: max number of log files before 
            rotating.
        @type max_num_logs: int
        @param logger_name: name by which this logger will be known.
        @type logger_name: str
        '''

        self._logging_level = logging_level
        self._log_file = logfile
        self.setup_logging(self._logging_level, 
                           self._log_file,
                           msg_identifier=msg_identifier,
                           rotating_logs=rotating_logs,
                           log_size=log_size,
                           max_num_logs=max_num_logs,
                           logger_name=logger_name)
        
        
    #-------------------------
    # loggingLevel
    #--------------
    
    @property
    def logging_level(self):
        return self._logging_level
        
    @logging_level.setter
    def logging_level(self, new_level):
        self._logging_level = new_level
        LoggingService.logger.setLevel(new_level)
        
    #-------------------------
    # logFile 
    #--------------
        
    @property
    def log_file(self):
        return self._log_file
    
    @log_file.setter
    def log_file(self, new_file):
        
        self._log_file = new_file
        # Remove the old logging handler.
        # Doesn't throw error if it's not there:
        LoggingService.logger.removeHandler(LoggingService.logging_handler)
        # Make a whole new logger with the
        # proper log file dest. This LoggingService 
        # instance will still be the same: 
        LoggingService.setup_logging(self.logging_level, 
                                     new_file, 
                                     LoggingService.msg_identifier, 
                                     LoggingService.rotating_logs, 
                                     LoggingService.log_size, 
                                     LoggingService.max_num_logs,
                                     logger_name=self.logger.name
                                     )

    #-------------------------
    # handlers 
    #--------------
    
    @property
    def handlers(self):
        return self.logger.handlers

    #-------------------------
    # setup_logging 
    #--------------
    
    @classmethod
    def setup_logging(cls, 
                      loggingLevel=logging.INFO, 
                      logFile=None,
                      msg_identifier=None,
                      rotating_logs=True,
                      log_size=1000000,
                      max_num_logs=500,
                      logger_name=None
                      ):
        '''
        Set up the standard Python logger.
        If msg_identifier is provided, it is shown
        at the very start of each logging message.
        If None, the file name of the calling module
        is used without the .py extension.

        @param loggingLevel: initial logging level
        @type loggingLevel: {logging.INFO|WARN|ERROR|DEBUG}
        @param logFile: optional file path where to send log entries
        @type logFile: str
        '''
        
        # Save parms in case someone changes the
        # logfile on this logger later:

        cls.msg_identifier = msg_identifier
        cls.rotating_logs  = rotating_logs
        cls.log_size       = log_size
        cls.max_num_logs   = max_num_logs
        
        # Make the name of the logger be the name
        # of this file, without the .py extension:
        if logger_name is None:
            (logger_name, _ext) = os.path.splitext(os.path.basename(__file__))
        LoggingService.logger = logging.getLogger(logger_name)

        # Create file handler if requested:
        if logFile is not None:
            if rotating_logs:
                # New log file every 10Mb, at most 500 times:
                handler = RotatingFileHandler(logFile,
                                              maxBytes=log_size,
                                              backupCount=max_num_logs
                                              )
            else:
                handler = logging.FileHandler(logFile)
                
            print('Logging of control flow will go to %s' % logFile)
        else:
            # Create console handler:
            handler = logging.StreamHandler()

        handler.setLevel(loggingLevel)

        # Create formatter
        #formatter = logging.Formatter("%(name)s: %(asctime)s;%(levelname)s: %(message)s")
        if msg_identifier is None:
            cls.msg_identifier = os.path.basename(sys.argv[0])

        formatter = logging.Formatter(f"{cls.msg_identifier}({os.getpid()}): %(asctime)s;%(levelname)s: %(message)s")

        handler.setFormatter(formatter)
        
        # Avoid double entries from the default logger:
        LoggingService.logger.propagate = False        

        # Add the handler to the logger
        LoggingService.logger.addHandler(handler)
        # Remember handler obj so we can remove it if 
        # a new logfile is requested:
        cls.logging_handler = handler
        LoggingService.logger.setLevel(loggingLevel)

    #-------------------------
    # log_debug/warn/info/err 
    #--------------

    def debug(self, msg):
        LoggingService.logger.debug(msg)

    def warn(self, msg):
        LoggingService.logger.warning(msg)

    def info(self, msg):
        LoggingService.logger.info(msg)

    def err(self, msg):
        LoggingService.logger.error(msg)

# ------------------------- Main ---------------

# For testing only; this module is intended for import.
if __name__ == '__main__':
    pass 
    
#    l = LoggingService()
#    print(l)
#    lsame = LoggingService()
#    print(lsame)
#     l1 = LoggingService(log_file='/tmp/trash.log')
#     print(l1)
#     l1same = LoggingService(log_file='/tmp/trash.log')
#     print(l1same)
#     l2 = LoggingService()
#     print(l2)

#     l = LoggingService(log_file='/tmp/trash.log',
#                        rotating_logs=True,
#                        log_size=10,
#                        max_num_logs=4
#                        )
#     l.info('123456789')
#     
    
