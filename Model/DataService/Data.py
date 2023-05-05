import time
import numpy as np
import pandas as pd
from config import config
from Model.LogService.Log import log_service
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataService:
    def __init__(self):
        self.data_set_path = config.dataset.dataset_path
        self.label_column = config.dataset.label_column_str
        self.train_size, self.test_size = map(int, config.dataset.train_test_split.split("-"))
        self.train_size, self.test_size = self.train_size / 100, self.test_size / 100

    def run(self) -> dict:
        """Main function of data service.

        Returns
        -------
        dict
            Dictionary that contains all the relevant information of the given data set.
        """
        start = time.time()

        results = {}

        results['Dataset'] = self.load_data(self.data_set_path)
        results['N. Dataset'] = self.normalize_features(results['Dataset'])

        train, test = self.train_test_split(results['N. Dataset'])
        results['Train'] = train
        results['Test'] = test

        results['Features'] = self.get_features(results['Train'][1])
        results['Labels'] = self.get_labels(results['Train'][2])

        end = time.time()
        log_service.log('Info', f'[Data Service] : Data set name: ({self.data_set_path}.csv) * Number of features: '
                                f'({len(results["Features"])}) * Number of labels: ({len(results["Labels"])})')
        log_service.log('Debug', f'[Data Service] : Total run time in seconds: [{round(end - start, 3)}]')
        return results

    @staticmethod
    def load_data(data_set: str) -> pd.DataFrame:
        """Load data.

        Parameters
        ----------
        data_set : str
            Dataset name

        Returns
        -------
        pandas.DataFrame
            Processed data as a data frame.
        """
        df = pd.DataFrame(pd.read_csv(f'./Model/Datasets/{data_set}.csv'))
        return df

    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize the feature values to [0, 1] range.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data

        Returns
        -------
        pandas.DataFrame
            Normalized input data.
        """
        column_names = X.columns
        cols_to_normalize = X.columns.difference([self.label_column])

        scaler = MinMaxScaler()

        X[cols_to_normalize] = scaler.fit_transform(X[cols_to_normalize])
        X.columns = column_names

        return X

    def train_test_split(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets

         Parameters
         ----------
         df : pandas.DataFrame
             Data frame of the data set

         Returns
         -------
         tuple
             Train and test data as two separate tuples.
         """
        train, test = train_test_split(df, test_size=self.test_size)

        X_train = train.drop(self.label_column, axis=1)
        y_train = pd.DataFrame(train[self.label_column])
        X_test = test.drop(self.label_column, axis=1)
        y_test = pd.DataFrame(test[self.label_column])

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
            Feature names of the given data set.
        """
        return X.columns

    def get_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract labels names

        Parameters
        ----------
        X : pandas.DataFrame
            Input data

        Returns
        -------
        pandas.DataFrame
            Label names of the given data set.
        """
        labels_names = X[self.label_column].unique()
        return labels_names

    def k_fold_split(self, data: pd.DataFrame, train_index: np.ndarray, val_index: np.ndarray) -> tuple:
        """Split data into train and validation sets

         Parameters
         ----------
         data : pandas.DataFrame
             Data frame of the data set
         train_index : numpy.ndarray
             Indexes associated with the training set (in the current fold)
         val_index : numpy.ndarray
             Indexes associated with the validation set (in the current fold)

         Returns
         -------
         tuple
             Train and validation data as two separate tuples.
         """
        train = data.iloc[train_index]
        X_train = train.drop(self.label_column, axis=1)
        y_train = pd.DataFrame(train[self.label_column])

        val = data.iloc[val_index]
        X_val = val.drop(self.label_column, axis=1)
        y_val = pd.DataFrame(val[self.label_column])

        return (train, X_train, y_train), (val, X_val, y_val)
