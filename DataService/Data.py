import time
import pandas as pd
from sklearn.model_selection import train_test_split


class DataService:
    def __init__(self, log_service):
        self.log_service = log_service

    def execute_data_service(self, data_set: str) -> dict:
        start = time.time()

        results = {}

        results['dataset'] = DataService.load_data(data_set)

        train, test = DataService.train_test_split(results['dataset'])
        results['train'] = train
        results['test'] = test

        results['features'] = DataService.get_features(results['train'][1])
        results['labels'] = DataService.get_labels(results['train'][2])

        end = time.time()
        self.log_service.log('Info', f'[Data Service] : Data set name: ({data_set}.csv) * Number of features: '
                                     f'({len(results["features"])}) * Number of labels: ({len(results["labels"])})')
        self.log_service.log('Debug', f'[Data Service] : Total run time in seconds: [{round(end-start, 3)}]')
        return results

    @staticmethod
    def load_data(data_set: str) -> pd.DataFrame:
        """Load data

        Parameters
        ----------
        data_set : pandas.DataFrame
            Input data

        Returns
        -------
        pandas.DataFrame
            Processed data as a data frame.
        """
        df = pd.DataFrame(pd.read_csv(f'./Datasets/{data_set}.csv'))
        return df

    @staticmethod
    def train_test_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> tuple:
        """Split data into train and test sets

         Parameters
         ----------
         df : pandas.DataFrame
             Data frame of the data set
         test_size : float, optional
             Test set size, by default 0.25
         random_state : int, optional
             Random seed for data shuffling, by default 42

         Returns
         -------
         tuple
             Train and test data as two separate tuples (X_train, y_train) and (X_test, y_test)
         """
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        X_train = train.drop('class', axis=1)
        y_train = pd.DataFrame(train['class'])
        X_test = test.drop('class', axis=1)
        y_test = pd.DataFrame(test['class'])

        return (train, X_train, y_train), (test, X_test, y_test)

    @staticmethod
    def get_features(X: pd.DataFrame) -> pd.DataFrame:
        """Extract features names

        Parameters
        ----------
        X : pandas.DataFrame
            Input data

        Returns
        -------
        pandas.DataFrame
            Features names of the given data set.
        """
        return X.columns

    @staticmethod
    def get_labels(X: pd.DataFrame) -> pd.DataFrame:
        """Extract labels names

        Parameters
        ----------
        X : pandas.DataFrame
            Input data

        Returns
        -------
        pandas.DataFrame
            Labels names of the given data set.
        """
        labels_names = X['class'].unique()
        return labels_names
