import math
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
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service
        self.classifiers = [LogisticRegression(),
                            tree.DecisionTreeClassifier(),
                            KNeighborsClassifier(n_neighbors=5)]
        self.cv = KFold(n_splits=10, random_state=41, shuffle=True)

    def classify(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, clustering_res: list, features: list,
                 n_values: int) -> dict:
        # Train
        K_values = [*range(2, len(features), 1)]
        train_res = self.execute_classification_service(X, y, F, clustering_res, features, K_values, 'Train')
        self.visualization_service.plot_accuracy_to_silhouette(train_res, clustering_res, 'Jeffries-Matusita', 'Train')

        # Test
        K_values = self.choose_K_test_values(train_res['Results By Accuracy'], len(features), n_values)
        test_res = self.execute_classification_service(X, y, F, clustering_res, features, K_values, 'Test')
        self.visualization_service.plot_accuracy_to_silhouette(test_res, clustering_res, 'Jeffries-Matusita', 'Test')

        return {
            "Train": train_res,
            "Test": test_res
        }

    def execute_classification_service(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, clustering_res: list,
                                       features: list, K_values: list, mode: str) -> dict:
        start = time.time()

        evaluations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit a task for each K value
            tasks = [executor.submit(self.execute_classification, X, y, F, clustering_res[K - 2], features, K, mode)
                     for K in K_values]

            # Wait for the tasks to complete and store the results
            for task in concurrent.futures.as_completed(tasks):
                evaluations.append(task.result())

        evaluations = self.arrange_results(evaluations)

        end = time.time()
        self.log_service.log(f'Debug, [Classification Service] - [{mode}]: Total run time in seconds:'
                             f' [{round(end - start, 3)}]')

        return evaluations

    def execute_classification(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, results: dict, features: list,
                               K: int, mode: str) -> dict:
        new_X = self.prepare_data(X, F, results['Kmedoids']['Centroids'], features)
        evaluation = self.evaluate(new_X, y, K, mode)
        return evaluation

    @staticmethod
    def prepare_data(X: pd.DataFrame, F: np.ndarray, centroids: np.ndarray, features: list) -> pd.DataFrame:
        features_indexes = [i for i in range(len(F)) if F[i] in centroids]
        features_names = [features[i] for i in features_indexes]
        new_X = X[X.columns.intersection(features_names)]

        return new_X

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, K: int, mode: str) -> dict:
        classifiers_str = []
        classifier_to_std, classifier_to_mean = {}, {}

        for classifier in self.classifiers:
            std_score, mean_score = 0, 0
            classifier_str = str(classifier).replace('(', '').replace(')', '')
            classifiers_str.append(classifier_str)

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
            'K': K,
            'Mode': mode,
            'Std': classifier_to_std,
            'Mean': classifier_to_mean,
            'Classifiers': classifiers_str
        }

    def arrange_results(self, results: list) -> dict:
        b_results = results
        for result in sorted(results, key=lambda x: x['K']):
            k = result['K']
            for classifier, classifier_res in result['Mean'].items():
                self.log_service.log('Info', f'[Classification Service] : Accuracy result for ({k}) features with '
                                             f'({classifier}) --> ({round(classifier_res, 3)})')

        new_results = {
            'Results By K': sorted(results, key=lambda x: x['K']),
            'Results By Accuracy': sorted(results, key=lambda x: sum(x['Mean'].values())/len(x['Mean']), reverse=True),
            'Results By Classifiers': self.arrange_results_by_classifier(b_results)
        }
        return new_results

    @staticmethod
    def arrange_results_by_classifier(results: list) -> dict:
        new_results = {}
        classifiers = list(results[0]['Classifiers'])

        for classifier in classifiers:
            classifiers_res = {}
            for result in results:
                K = result['K']
                classification_results = result['Mean']
                for c, res in classification_results.items():
                    if c == classifier:
                        classifiers_res[K] = res
            new_results[classifier] = dict(sorted(classifiers_res.items(), key=lambda item: item[0]))

        return new_results

    def choose_K_test_values(self, train_res: dict, n_features: int, n_values: int) -> list:
        # Default values
        K_values = [# n_features,
                    math.ceil((1 / 4) * n_features),
                    math.ceil((1 / 8) * n_features),
                    math.ceil((1 / 16) * n_features)]
        if n_features > 32:
            K_values.append(math.ceil((1 / 32) * n_features))

        # Best 'n_values' K values by accuracy
        K_values += [int(result['K']) for result in train_res if int(result['K']) not in K_values][:n_values]
        K_values = sorted(K_values)
        return K_values