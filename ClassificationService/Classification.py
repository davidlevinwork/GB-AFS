import time
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


NUMBER_OF_TEST_EPOCHS = 1
NUMBER_OF_TRAIN_EPOCHS = 10


class ClassificationService:
    def __init__(self, log_service):
        self.log_service = log_service
        self.classifiers = [LogisticRegression(),
                            tree.DecisionTreeClassifier(),
                            KNeighborsClassifier(n_neighbors=5)]
        self.cv = KFold(n_splits=10, random_state=41, shuffle=True)

    def execute_classification_service(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, results: list,
                                       features: list, n_features: int, mode: str) -> list:
        start = time.time()
        K_values = [*range(2, n_features, 1)]

        evaluations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit a task for each K value
            tasks = [executor.submit(self.execute_classification, X, y, F, results[K-2], features, K, mode)
                     for K in K_values]

            # Wait for the tasks to complete and store the results
            for task in concurrent.futures.as_completed(tasks):
                evaluations.append(task.result())

        end = time.time()
        self.log_service.log('Debug', f'[Classification Service] : Total run time in seconds: [{round(end-start, 3)}]')

        return evaluations

    def execute_classification(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, results: dict, features: list,
                               K: int, mode: str) -> dict:
        new_X = self.prepare_data(X, F, results['kmedoids']['centroids'], features)
        evaluation = self.evaluate(new_X, y, K, mode)
        return evaluation

    @staticmethod
    def prepare_data(X: pd.DataFrame, F: np.ndarray, centroids: np.ndarray, features: list) -> pd.DataFrame:
        features_indexes = [i for i in range(len(F)) if F[i] in centroids]
        features_names = [features[i] for i in features_indexes]
        new_X = X[X.columns.intersection(features_names)]

        return new_X

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, K: int, mode: str) -> dict:
        classifier_to_std, classifier_to_mean = {}, {}

        for classifier in self.classifiers:
            std_score, mean_score = 0, 0
            classifier_str = str(classifier).replace('(', '').replace(')', '')

            epochs = NUMBER_OF_TRAIN_EPOCHS if mode == 'Train' else NUMBER_OF_TEST_EPOCHS
            for _ in range(epochs):
                scores = cross_val_score(classifier, X, np.ravel(y), scoring='accuracy', cv=self.cv, n_jobs=1)
                std_score += np.std(scores)
                mean_score += np.mean(scores)

            std_score /= epochs
            mean_score /= epochs

            classifier_to_std[classifier_str] = std_score
            classifier_to_mean[classifier_str] = mean_score

        return {
            'k': K,
            'mode': mode,
            'std': classifier_to_std,
            'mean': classifier_to_mean
        }
