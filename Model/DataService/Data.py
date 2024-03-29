import time
import random
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
        """
        Main function of data service.

        Returns:
            dict: A dictionary that contains all the relevant information of the given data set.
                  Keys include 'Dataset', 'N. Dataset', 'Train', 'Test', 'Features', and 'Labels'.
        """
        start_time = time.time()

        dataset = self._load_data(self.data_set_path)
        normalized_dataset = self._normalize_features(dataset)

        train, test = self._train_test_split(normalized_dataset)

        results = {
            'Dataset': dataset,
            'N. Dataset': normalized_dataset,
            'Train': train,
            'Test': test,
            'Features': self._get_features(train[1]),
            'Labels': self._get_labels(train[2]),
            'Costs': self._generate_feature_costs(self._get_features(train[1]))
        }

        end_time = time.time()
        log_service.log('Info', f'[Data Service] : Data set path: ({self.data_set_path}.csv) *'
                                f'Number of features: ({len(results["Features"])}) *'
                                f' Number of labels: ({len(results["Labels"])})')
        log_service.log('Debug', f'[Data Service] : Total run time in seconds: [{round(end_time - start_time, 3)}]')
        return results

    @staticmethod
    def _load_data(data_set: str) -> pd.DataFrame:
        """
        Load the data.

        Args:
            data_set (str): Dataset name

        Returns:
            pd.DataFrame: Processed data as a data frame.
        """
        return pd.read_csv(data_set)

    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the feature values to [0, 1] range.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Normalized input data.
        """
        cols_to_normalize = X.columns.difference([self.label_column])
        scaler = MinMaxScaler()
        X[cols_to_normalize] = scaler.fit_transform(X[cols_to_normalize])

        return X

    @staticmethod
    def _generate_feature_costs(features: pd.DataFrame) -> dict:
        """
        Generate costs for each feature in the given range.

        Args:
            features (pd.DataFrame): Feature names of the given data set.

        Returns:
            dict: A dictionary containing the costs of each feature.
        """
        return {feature: random.uniform(0, 2) for feature in features}

    def _train_test_split(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train and test sets.

        Args:
            df (pd.DataFrame): Data frame of the data set

        Returns:
            tuple: Train and test data as two separate tuples.
        """
        train, test = train_test_split(df, test_size=self.test_size)

        X_train = train.drop(self.label_column, axis=1)
        y_train = pd.DataFrame(train[self.label_column])
        X_test = test.drop(self.label_column, axis=1)
        y_test = pd.DataFrame(test[self.label_column])

        return (train, X_train, y_train), (test, X_test, y_test)

    @staticmethod
    def _get_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature names.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Feature names of the given data set.
        """
        return X.columns

    def _get_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract label names.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Label names of the given data set.
        """
        return X[self.label_column].unique()

    def k_fold_split(self, data: pd.DataFrame, train_index: np.ndarray, val_index: np.ndarray) -> tuple:
        """
        Split data into train and validation sets.

        Args:
            data (pd.DataFrame): Data frame of the data set
            train_index (np.ndarray): Indexes associated with the training set (in the current fold)
            val_index (np.ndarray): Indexes associated with the validation set (in the current fold)

        Returns:
            tuple: Train and validation data as two separate tuples.
        """
        train = data.iloc[train_index]
        X_train = train.drop(self.label_column, axis=1)
        y_train = pd.DataFrame(train[self.label_column])

        val = data.iloc[val_index]
        X_val = val.drop(self.label_column, axis=1)
        y_val = pd.DataFrame(val[self.label_column])

        return (train, X_train, y_train), (val, X_val, y_val)
