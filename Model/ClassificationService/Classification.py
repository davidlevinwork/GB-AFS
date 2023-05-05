import time
import numpy as np
import pandas as pd
from sklearn import tree
import concurrent.futures
from Model.LogService.Log import log_service
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


NUMBER_OF_TEST_EPOCHS = 1
NUMBER_OF_TRAIN_EPOCHS = 10


class ClassificationService:
    def __init__(self):
        self.classifiers = [tree.DecisionTreeClassifier(),
                            KNeighborsClassifier(n_neighbors=5),
                            RandomForestClassifier()]
        self.cv = KFold(n_splits=10, shuffle=True)

    def classify(self, mode: str, data: dict, F: np.ndarray, clustering_res: list, features: list,
                 K_values: list) -> dict:
        """
        This function trains and validates the classification models using the provided data, feature matrix,
        clustering results, features, and K values. The function supports two modes: 'Train' and 'Full Train'.

        Args:
            mode (str): The mode of the classification service, either 'Train' or 'Full Train'.
            data (dict): A dictionary containing the training and validation data (if mode is 'Train') or only the
                         training data (if mode is 'Full Train').
            F (np.ndarray): The feature matrix.
            clustering_res (list): A list of clustering results.
            features (list): A list of feature names.
            K_values (list): A list of K values for K-medoids clustering.

        Returns:
            dict: A dictionary containing the classification results for the training and validation data (if mode
                  is 'Train') or only the training data (if mode is 'Full Train').

        """
        if mode == 'Train':
            # Train
            X, y = data['Train'][0], data['Train'][1]
            train_res = self.execute_classification_service(X, y, F, clustering_res, features, K_values, 'Train')

            # Validation
            X, y = data['Validation'][0], data['Validation'][1]
            test_res = self.execute_classification_service(X, y, F, clustering_res, features, K_values, 'Validation')

            results = {"Train": train_res, "Test": test_res}

        elif mode == 'Full Train':
            # (All) Train
            X, y = data['Train'][0], data['Train'][1]
            train_res = self.execute_classification_service(X, y, F, clustering_res, features, K_values, 'Full Train')

            results = {"Train": train_res}

        return results

    def execute_classification_service(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, clustering_res: list,
                                       features: list, K_values: list, mode: str) -> dict:
        """
        This function runs the classification service for the given data, feature matrix, clustering results,
        features, and K values in the specified mode.

        Args:
            X (pd.DataFrame): The input data as a DataFrame.
            y (pd.DataFrame): The target labels as a DataFrame.
            F (np.ndarray): The feature matrix.
            clustering_res (list): A list of clustering results.
            features (list): A list of feature names.
            K_values (list): A list of K values for K-medoids clustering.
            mode (str): The mode of the classification service, either 'Train', 'Validation', or 'Full Train'.

        Returns:
            dict: A dictionary containing the classification results.
        """
        start = time.time()

        evaluations = []
        if mode == 'Train' or mode == 'Validation':
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit a task for each K value
                tasks = [executor.submit(self.execute_classification, X, y, F, clustering_res[K - 2], features, K, mode)
                         for K in K_values]

                # Wait for the tasks to complete and store the results
                for task in concurrent.futures.as_completed(tasks):
                    evaluations.append(task.result())
        else:
            for K in K_values:
                K_clustering = [x for x in clustering_res if x['K'] == K][0]
                evaluations.append(
                    self.execute_classification(X, y, F, K_clustering, features, K, mode)
                )

        evaluations = self.arrange_results(evaluations)

        end = time.time()
        log_service.log('Debug', f'[Classification Service] - [{mode}]: Total run time in seconds:'
                                 f' [{round(end - start, 3)}]')

        return evaluations

    def execute_classification(self, X: pd.DataFrame, y: pd.DataFrame, F: np.ndarray, results: dict, features: list,
                               K: int, mode: str) -> dict:
        """
        This function executes the classification for the given data, feature matrix, clustering results, features,
        and K value in the specified mode.

        Args:
            X (pd.DataFrame): The input data as a DataFrame.
            y (pd.DataFrame): The target labels as a DataFrame.
            F (np.ndarray): The feature matrix.
            results (dict): A dictionary containing the clustering results for the current K value.
            features (list): A list of feature names.
            K (int): The current K value for K-medoids clustering.
            mode (str): The mode of the classification service, either 'Train', 'Validation', or 'Full Train'.

        Returns:
            dict: A dictionary containing the classification results for the current K value.
        """
        new_X = self.prepare_data(X, F, results['Kmedoids']['Centroids'], features)
        evaluation = self.evaluate(new_X, y, K, mode)

        log_service.log('Debug', f'[Classification Service] - Finished to execute classification for {K} '
                                 f'in mode {mode}.')

        return evaluation

    @staticmethod
    def prepare_data(X: pd.DataFrame, F: np.ndarray, centroids: np.ndarray, features: list) -> pd.DataFrame:
        """
        This function prepares the input data by selecting the relevant features based on the provided feature matrix
        and centroids.

        Args:
            X (pd.DataFrame): The input data as a DataFrame.
            F (np.ndarray): The feature matrix.
            centroids (np.ndarray): The centroids of the clustering results.
            features (list): A list of feature names.

        Returns:
            pd.DataFrame: The prepared input data as a DataFrame.
        """
        features_indexes = [i for i in range(len(F)) if F[i] in centroids]
        features_names = [features[i] for i in features_indexes]
        new_X = X[X.columns.intersection(features_names)]

        return new_X

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, K: int, mode: str) -> dict:
        """
        This function evaluates the classification models using the prepared input data and target labels for the
        specified K value and mode. The evaluation metrics include accuracy, F1-score, and AUC (both 'ovo' and 'ovr').

        Args:
            X (pd.DataFrame): The prepared input data as a DataFrame.
            y (pd.DataFrame): The target labels as a DataFrame.
            K (int): The current K value for K-medoids clustering.
            mode (str): The mode of the classification service, either 'Train', 'Validation', or 'Full Train'.

        Returns:
            dict: A dictionary containing the evaluation results for the classification models.
        """
        classifiers_str = []
        classifier_to_accuracy = {}
        classifier_to_f1 = {}
        classifier_to_auc_ovo = {}
        classifier_to_auc_ovr = {}

        lb = LabelBinarizer()
        y_binary = lb.fit_transform(y)  # Convert true labels to binary format
        for classifier in self.classifiers:
            accuracy_list, f1_list, auc_ovo_list, auc_ovr_list = [], [], [], []
            classifier_str = str(classifier).replace('(', '').replace(')', '')
            classifiers_str.append(classifier_str)

            epochs = NUMBER_OF_TRAIN_EPOCHS if mode == 'Train' else NUMBER_OF_TEST_EPOCHS
            for _ in range(epochs):
                cv_predictions = cross_val_predict(classifier, X, np.ravel(y), cv=self.cv)
                cv_predictions_proba = cross_val_predict(classifier, X, np.ravel(y), cv=self.cv, method='predict_proba')
                accuracy = accuracy_score(y, cv_predictions)
                f1 = f1_score(y, cv_predictions, average='weighted')
                auc_ovo = roc_auc_score(y_binary, cv_predictions_proba, multi_class='ovo', average='weighted')
                auc_ovr = roc_auc_score(y_binary, cv_predictions_proba, multi_class='ovr', average='weighted')

                f1_list.append(f1)
                accuracy_list.append(accuracy)
                auc_ovo_list.append(auc_ovo)
                auc_ovr_list.append(auc_ovr)

            classifier_to_f1[classifier_str] = np.mean(f1_list)
            classifier_to_accuracy[classifier_str] = np.mean(accuracy_list)
            classifier_to_auc_ovo[classifier_str] = np.mean(auc_ovo_list)
            classifier_to_auc_ovr[classifier_str] = np.mean(auc_ovr_list)

        return {
            'K': K,
            'Mode': mode,
            'Classifiers': classifiers_str,
            'F1': classifier_to_f1,
            'AUC-ovo': classifier_to_auc_ovo,
            'Accuracy': classifier_to_accuracy,
            'AUC-ovr': classifier_to_auc_ovr
        }

    def arrange_results(self, results: list) -> dict:
        """
        This function arranges the classification results in various ways, such as by K, classifiers, F1-score,
        AUC ('ovo' and 'ovr'), and accuracy.

        Args:
            results (list): A list of classification results.

        Returns:
            dict: A dictionary containing the arranged classification results.
        """
        b_results = results

        new_results = {
            'Results By K': sorted(results, key=lambda x: x['K']),
            'Results By Classifiers': self.arrange_results_by_classifier(b_results),
            'Results By F1': sorted(results, key=lambda x: sum(x['F1'].values()) / len(x['F1']), reverse=True),
            'Results By AUC-ovo': sorted(results, key=lambda x: sum(x['AUC-ovo'].values()) / len(x['AUC-ovo']),
                                         reverse=True),
            'Results By AUC-ovr': sorted(results, key=lambda x: sum(x['AUC-ovr'].values()) / len(x['AUC-ovr']),
                                         reverse=True),
            'Results By Accuracy': sorted(results, key=lambda x: sum(x['Accuracy'].values()) / len(x['Accuracy']),
                                          reverse=True)
        }
        return new_results

    @staticmethod
    def arrange_results_by_classifier(results: list) -> dict:
        """
        This function arranges the classification results by classifiers.

        Args:
            results (list): A list of classification results.

        Returns:
            dict: A dictionary containing the classification results arranged by classifiers.
        """
        new_results = {}
        classifiers = list(results[0]['Classifiers'])

        for classifier in classifiers:
            classifiers_res = {}
            for result in results:
                K = result['K']
                classification_results = result['Accuracy']
                for c, res in classification_results.items():
                    if c == classifier:
                        classifiers_res[K] = res
            new_results[classifier] = dict(sorted(classifiers_res.items(), key=lambda item: item[0]))

        return new_results
