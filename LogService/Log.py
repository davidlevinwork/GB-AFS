import os
import re
import glob
import shutil
from datetime import datetime

log_file_name = os.path.join(os.getcwd(), "Files", "Log.txt")


class LogService:
    def __init__(self):
        try:
            open(log_file_name, 'w').close()
        except OSError as ex:
            print(f'Failed to creat log file. Error: [{ex}]')
        else:
            print('Log File created successfully.')

    @staticmethod
    def log(level="Info", data=None):
        # level = debug, info, warning, error, critical
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        str_format = "[{0: ^21}] | [{1:^9}] : {2}\n".format(date_str, level, data)
        try:
            log_file = open(log_file_name, 'a')
            log_file.write(str_format)
            log_file.close()
        except OSError as ex:
            print(f'The log file doesn\'t exist!. Error: [{ex}]')