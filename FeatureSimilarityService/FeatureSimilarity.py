import math
import numpy as np
import pandas as pd
from FeatureSimilarityService.Distances import get_distance


class FeatureSimilarityService:
    def __init__(self, log_service):
        self.log_service = log_service

    @staticmethod
    def calculate_separation_matrix(X: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, distance_measure: str) -> np.ndarray:
        label_combinations = FeatureSimilarityService.get_label_combinations(labels)
        separation_matrix = FeatureSimilarityService.init_feature_separation_matrix(features, labels)

        for i, feature in enumerate(features):                                      # Iterate over the features
            for j, labels in enumerate(label_combinations):                         # Iterate over each pairs of classes
                separation_matrix[i][j] = get_distance(distance_measure, X, feature, labels[0], labels[1])
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