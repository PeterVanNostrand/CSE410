# import os
from enum import IntEnum

class LOGLEVEL(IntEnum):
    NOLOG = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

class Logger():
    def __init__(self, log_level, file_path):
        self.log_level = log_level
        self.log_file = open(file_path, "a")
        print("###### NEW LOG SESSION ######", file=self.log_file)

    def log(self, message, level=LOGLEVEL.DEBUG, console=False):
        if level <= self.log_level: 
            print(message, file=self.log_file)
            if console:
                print(message)