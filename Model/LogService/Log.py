import os
from datetime import datetime


class LogService:
    def __init__(self):
        try:
            open(log_file_name, 'w').close()
        except OSError as ex:
            print(f'Failed to create log file. Error: [{ex}]')
        else:
            print('Log File created successfully.')

    @staticmethod
    def log(level="Info", data=None):
        """Log a message with the given log level and data.

        Args:
            level (str, optional): Log level, one of "Debug", "Info", "Warning", "Error", "Critical".
            data (str, optional): The message to log.
        """
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        str_format = "[{0: ^21}] | [{1:^9}] : {2}\n".format(date_str, level, data)

        try:
            with open(log_file_name, 'a') as log_file:
                log_file.write(str_format)
        except OSError as ex:
            print(f"The log file doesn't exist! Error: [{ex}]")


# Create a global log_service instance
log_file_name = os.path.join(os.getcwd(), "Model", "Files", "Log.txt")
log_service = LogService()
