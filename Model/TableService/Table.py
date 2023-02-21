import os
import glob
from prettytable import PrettyTable


class TableService:
    @staticmethod
    def create_table(fold_index: str, stage: str, classification_res: dict):
        test_results = classification_res['Test']['Results By K']
        train_results = classification_res['Train']['Results By K']

        headers = ['K Value'] + classification_res['Train']['Results By K'][0]['Classifiers']
        seperator = ['*'] * len(headers)
        table = PrettyTable([header for header in headers])

        for train_result in train_results:
            row = [train_result['K']] + list(train_result['Mean'].values())
            table.add_row([col for col in row])

        table.add_row([sep for sep in seperator])

        for test_result in test_results:
            row = [test_result['K']] + list(test_result['Mean'].values())
            table.add_row([col for col in row])

        TableService.save_table(fold_index, stage, table)

    @staticmethod
    def save_table(fold_index, stage, table):
        try:
            current_dir = os.path.dirname(__file__)
            plots_dir = os.path.join(current_dir, '..', 'Files', 'Plots')
            latest_plot_dir = max(glob.glob(os.path.join(plots_dir, '*/')), key=os.path.getmtime).rsplit('\\', 1)[0]

            if stage == "Train":
                table_file = os.path.join(latest_plot_dir, stage, f'Fold #{fold_index}', 'Results.txt')
            else:
                table_file = os.path.join(latest_plot_dir, stage, 'Results.txt')

            with open(table_file, 'w') as w:
                data = table.get_string(title="Classification Results")
                w.write(data)
            w.close()
        except OSError as ex:
            print(f'Failed to save results table as a file. Error: [{ex}]')
