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
    def __init__(self, file_path, log_level=LOGLEVEL.DEBUG, write_mode="a"):
        self.log_level = log_level
        self.log_file = open(file_path, write_mode)
        print("###### NEW LOG SESSION ######", file=self.log_file)

    def log(self, message, level=LOGLEVEL.DEBUG, console=False, end="\n"):
        if level <= self.log_level:
            print(message, file=self.log_file, end=end)
            if console:
                print(message, end=end)
