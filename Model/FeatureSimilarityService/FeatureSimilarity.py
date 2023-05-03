import math
import time
import numpy as np
import pandas as pd
from Model.FeatureSimilarityService.Distances import get_distance

from Model.LogService.Log import log_service


class FeatureSimilarityService:
    @staticmethod
    def calculate_JM_matrix(X: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
        """Calculate the JM matrix

        Parameters
        ----------
        X : pandas.DataFrame
            Data frame of the data set
        features: pandas.DataFrame
            Feature names of the given data set
        labels : pandas.DataFrame
            Label names of the given data set

        Returns
        -------
        np.ndarray
            JM (feature similarity) matrix.
        """
        start = time.time()
        label_combinations = FeatureSimilarityService.get_label_combinations(labels)
        separation_matrix = FeatureSimilarityService.init_JM_matrix(features, labels)

        for i, feature in enumerate(features):  # Iterate over the features
            for j, labels in enumerate(label_combinations):  # Iterate over each pairs of classes
                separation_matrix[i][j] = get_distance(X, feature, labels[0], labels[1])
            log_service.log('Info', f'[Feature Similarity Service] : Finish to compute separation value of '
                                    f'feature ({feature}), index ({i + 1})')

        end = time.time()
        log_service.log('Debug', f'[Feature Similarity Service] : Total run time in seconds: '
                                 f'[{round(end - start, 3)}]')
        return separation_matrix

    @staticmethod
    def get_label_combinations(labels: pd.DataFrame) -> list:
        """Get all the possible combinations of the given labels

        Parameters
        ----------
        labels : pandas.DataFrame
            Label names of the given data set

        Returns
        -------
        list
            list of all the label combinations.
        """
        combinations = []
        min_label, max_label = min(labels), max(labels)

        for i_label in range(min_label, max_label + 1):
            for j_label in range(i_label + 1, max_label + 1):
                combinations.append((i_label, j_label))
        return combinations

    @staticmethod
    def init_JM_matrix(features: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
        """Init an empty JM matrix

        Parameters
        ----------
        features: pandas.DataFrame
            Feature names of the given data set
        labels : pandas.DataFrame
            Label names of the given data set

        Returns
        -------
        np.ndarray
            New JM (feature similarity) matrix.
        """
        matrix = np.zeros((len(features), math.comb(len(labels), 2)))
        return matrix
