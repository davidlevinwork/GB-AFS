import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def execute_data_service(self, data_set: str) -> dict:
        """Main function of data service.

        Parameters
        ----------
        data_set : str
            Data set name

        Returns
        -------
        dict
            Dictionary that contains all the relevant information of the given data set.
        """
        start = time.time()

        results = {}

        results['Dataset'] = DataService.load_data(data_set)
        results['N. Dataset'] = DataService.normalize_features(results['Dataset'])

        train, test = DataService.train_test_split(results['N. Dataset'])
        results['Train'] = train
        results['Test'] = test

        results['Features'] = DataService.get_features(results['Train'][1])
        results['Labels'] = DataService.get_labels(results['Train'][2])

        end = time.time()
        self.log_service.log('Info', f'[Data Service] : Data set name: ({data_set}.csv) * Number of features: '
                                     f'({len(results["Features"])}) * Number of labels: ({len(results["Labels"])})')
        self.log_service.log('Debug', f'[Data Service] : Total run time in seconds: [{round(end-start, 3)}]')
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
        df = pd.DataFrame(pd.read_csv(f'./Datasets/{data_set}.csv'))
        return df

    @staticmethod
    def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
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
        cols_to_normalize = X.columns.difference(['class'])

        scaler = MinMaxScaler()

        X[cols_to_normalize] = scaler.fit_transform(X[cols_to_normalize])
        X.columns = column_names

        return X

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
             Train and test data as two separate tuples.
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
            Feature names of the given data set.
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
            Label names of the given data set.
        """
        labels_names = X['class'].unique()
        return labels_names

    @staticmethod
    def k_fold_split(data: pd.DataFrame, train_index: np.ndarray, val_index: np.ndarray) -> tuple:
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
        X_train = train.drop('class', axis=1)
        y_train = pd.DataFrame(train['class'])

        val = data.iloc[val_index]
        X_val = val.drop('class', axis=1)
        y_val = pd.DataFrame(val['class'])

        return (train, X_train, y_train), (val, X_val, y_val)

    @staticmethod
    def get_train_results(classification_results: dict, clustering_results: list, n_folds: int) -> dict:
        clustering = DataService.get_train_clustering(clustering_results)
        classification = DataService.get_train_classification(classification_results, n_folds)

        return {
            'Clustering': clustering,
            'Classification': classification
        }

    @staticmethod
    def get_train_classification(results: dict, n_folds: int) -> dict:
        # Init with the results of the first fold
        combined_results = results[0]['Test']['Results By Classifiers']

        # Sum
        for i in range(1, n_folds):
            classifiers = results[i]['Test']['Results By Classifiers']
            for classifier, classifier_results in classifiers.items():
                combined_results[classifier] = dict(Counter(combined_results[classifier]) + Counter(classifier_results))

        # Divide
        for classifier, classifier_results in combined_results.items():
            combined_results[classifier] = [x / n_folds for x in list(combined_results[classifier].values())]

        return combined_results

    @staticmethod
    def get_train_clustering(results: list) -> list:
        # Init with the results of the first fold
        combined_results = results[0]

        # Sum
        for i in range(1, len(results)):
            for j, result in enumerate(results[i]):
                k = result['K']
                sub_results = result['Silhouette']
                for sil_name, sil_value in sub_results.items():
                    combined_results[j]['Silhouette'][sil_name] += sil_value

        # Divide
        for result in combined_results:
            sub_results = result['Silhouette']
            for sil_name, sil_value in sub_results.items():
                sub_results[sil_name] /= len(results)

        return combined_results
