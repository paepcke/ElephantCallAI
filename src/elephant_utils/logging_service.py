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
                 rotating_logs=True,
                 log_size=1000000,
                 max_num_logs=500):

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
        @type logfile: str
        @param rotating_logs: whether or not to rotate logs. 
        @type rotating_logs: bool
        @param log_size: max size of each log file, if rotating 
        @type log_size: int
        @param max_num_logs: max number of log files before 
            rotating.
        @type max_num_logs: int
        '''

        self._logging_level = logging_level
        self._log_file = logfile
        self.setup_logging(self._logging_level, 
                           self._log_file,
                           rotating_logs,
                           log_size,
                           max_num_logs)
        
        
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
        #***** Should change the file. But no time for this now
        self._log_file = new_file

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
                      rotating_logs=True,
                      log_size=1000000,
                      max_num_logs=500                      
                      ):
        '''
        Set up the standard Python logger.

        @param loggingLevel: initial logging level
        @type loggingLevel: {logging.INFO|WARN|ERROR|DEBUG}
        @param logFile: optional file path where to send log entries
        @type logFile: str
        '''

        LoggingService.logger = logging.getLogger(os.path.basename(__file__))

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
        formatter = logging.Formatter("%(name)s: %(asctime)s;%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        
        # Avoid double entries from the default logger:
        LoggingService.logger.propagate = False        

        # Add the handler to the logger
        LoggingService.logger.addHandler(handler)
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
    
