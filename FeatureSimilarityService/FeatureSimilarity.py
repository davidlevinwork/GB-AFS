import math
import time
import numpy as np
import pandas as pd
from FeatureSimilarityService.Distances import get_distance


class FeatureSimilarityService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def calculate_separation_matrix(self, X: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame,
                                    distance_measure: str) -> np.ndarray:
        start = time.time()
        label_combinations = FeatureSimilarityService.get_label_combinations(labels)
        separation_matrix = FeatureSimilarityService.init_feature_separation_matrix(features, labels)

        for i, feature in enumerate(features):                                      # Iterate over the features
            for j, labels in enumerate(label_combinations):                         # Iterate over each pairs of classes
                separation_matrix[i][j] = get_distance(distance_measure, X, feature, labels[0], labels[1])
            self.log_service.log('Info', f'[Feature Similarity Service] : Finish to compute separation value of '
                                         f'feature ({feature}), index ({i + 1})')

        end = time.time()
        self.log_service.log('Debug', f'[Feature Similarity Service] : Total run time in seconds: '
                                      f'[{round(end-start, 3)}]')
        return separation_matrix

    @staticmethod
    def get_label_combinations(labels: pd.DataFrame) -> list:
        combinations = []
        min_label, max_label = min(labels), max(labels)

        for i_label in range(min_label, max_label + 1):
            for j_label in range(i_label + 1, max_label + 1):
                combinations.append((i_label, j_label))
        return combinations

    @staticmethod
    def init_feature_separation_matrix(features: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
        matrix = np.zeros((len(features), math.comb(len(labels), 2)))
        return matrix
