import os
import glob
from prettytable import PrettyTable


class TableService:
    @staticmethod
    def create_table(fold_index: str, stage: str, classification_res: dict):
        train_results = test_results = None

        metrics = ['F1', 'Recall', 'Accuracy', 'Precision', 'Specificity']

        for metric in metrics:
            if 'Train' in classification_res:
                train_results = classification_res['Train']['Results By K']
                headers = ['K Value'] + classification_res['Train']['Results By K'][0]['Classifiers']
            if 'Test' in classification_res:
                test_results = classification_res['Test']['Results By K']
                headers = ['K Value'] + classification_res['Test']['Results By K'][0]['Classifiers']

            seperator = ['*'] * len(headers)
            table = PrettyTable([header for header in headers])

            if train_results is not None:
                for train_result in train_results:
                    row = [train_result['K']] + list(train_result[metric].values())
                    table.add_row([col for col in row])

            if train_results is not None and test_results is not None:
                table.add_row([sep for sep in seperator])

            if test_results is not None:
                for test_result in test_results:
                    row = [test_result['K']] + list(test_result[metric].values())
                    table.add_row([col for col in row])

            TableService.save_table(table, metric, fold_index, stage)

    @staticmethod
    def save_table(table: PrettyTable, metric: str, fold_index: str, stage: str):
        try:
            current_dir = os.path.dirname(__file__)
            plots_dir = os.path.join(current_dir, '..', 'Files', 'Plots')
            latest_plot_dir = max(glob.glob(os.path.join(plots_dir, '*/')), key=os.path.getmtime).rsplit('\\', 1)[0]

            if stage == "Train":
                table_file = os.path.join(latest_plot_dir, stage, f'Fold #{fold_index}', f'{metric} Results.txt')
            else:
                dir = os.path.join(latest_plot_dir, stage)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                table_file = os.path.join(latest_plot_dir, stage, f'{metric} Results.txt')

            with open(table_file, 'w') as w:
                data = table.get_string(title=f"Classification Results - {metric}")
                w.write(data)
            w.close()
        except OSError as ex:
            print(f'Failed to save results table as a file. Error: [{ex}]')
