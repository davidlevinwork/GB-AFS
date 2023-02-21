from Model.Executor import Executor

if __name__ == '__main__':
    executor = Executor.Executor()
    executor.init_services()
    executor.execute()
