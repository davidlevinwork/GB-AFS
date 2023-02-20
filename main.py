from Executor import Executor
from ConfigurationService import ConfigurationManager

if __name__ == '__main__':
    executor = Executor.Executor(ConfigurationManager.ConfigManager('config.json'))
    executor.init_services()
    executor.execute()
