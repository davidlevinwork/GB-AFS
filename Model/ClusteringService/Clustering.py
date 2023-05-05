import time
import numpy as np
import pandas as pd
from array import array
from config import config
import concurrent.futures
from sklearn_extra.cluster import KMedoids
from Model.LogService.Log import log_service
from sklearn.metrics import silhouette_score
from Model.ClusteringService.Silhouette import simplified_silhouette
from Model.VisualizationService.Visualization import visualization_service


class ClusteringService:
    def execute_clustering_service(self, F: np.ndarray, K_values: list, stage: str, fold_index: int) -> list:
        """
        This function executes the clustering service for the given low-dimensional feature matrix, list of K values,
        stage, and fold index.

        Args:
            F (np.ndarray): The low-dimensional feature matrix.
            K_values (list): A list of K values to test.
            stage (str): The stage of the algorithm ('Train', 'Full Train', 'Test').
            fold_index (int): The index of the given k-fold (used for saving results & plots).

        Returns:
            list: A list of clustering and silhouette results.
        """
        start = time.time()

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit a task for each K value
            tasks = [executor.submit(ClusteringService.execute_clustering, F, K) for K in K_values]

            # Wait for the tasks to complete and store the results
            for task in concurrent.futures.as_completed(tasks):
                results.append(task.result())

        results = self.arrange_results(results)
        visualization_service.plot_silhouette(results, stage, fold_index)
        visualization_service.plot_clustering(F, results, stage, fold_index)
        visualization_service.plot_upgrade_clustering(F, results, stage, fold_index)

        end = time.time()
        log_service.log('Debug', f'[Clustering Service] : Total run time in seconds: [{round(end - start, 3)}]')
        return results

    @staticmethod
    def execute_clustering(F: pd.DataFrame, K: int) -> dict:
        """
        This function executes the K-Medoid clustering algorithm for the given low-dimensional feature matrix and
        K value.

        Args:
            F (pd.DataFrame): The low-dimensional feature matrix.
            K (int): The number of clusters.

        Returns:
            dict: A dictionary containing K-Medoids and silhouette results for the given K.
        """
        results = {}

        results['K'] = K
        results['Kmedoids'] = ClusteringService.run_kmedoids(F, K)
        results['Silhouette'] = ClusteringService.calculate_silhouette_value(X=F,
                                                                             y=results['Kmedoids']['Labels'],
                                                                             centroids=results['Kmedoids']['Centroids'])
        return results

    @staticmethod
    def run_kmedoids(F: pd.DataFrame, K: int, init: str = 'k-medoids++') -> dict:
        """
        This function performs K-medoids clustering on the given feature matrix and K value. It also accepts optional
         arguments for the method and initialization of medoids.

        Args:
            F (pd.DataFrame): The reduced feature similarity matrix.
            K (int): The number of clusters.
            init (str, optional): The medoid initialization method ('random', 'heuristic', 'k-medoids++', 'build').
                                  Defaults to 'k-medoids++'.

        Returns:
            dict: A dictionary containing cluster labels and centroids.
        """
        kmedoids = KMedoids(init=init,
                            n_clusters=K,
                            method=config.k_medoids.method).fit(F)
        results = {
            'Labels': kmedoids.labels_,
            'Features': kmedoids.medoid_indices_,
            'Centroids': kmedoids.cluster_centers_
        }
        return results

    @staticmethod
    def calculate_silhouette_value(X: pd.DataFrame, y: pd.DataFrame, centroids: array) -> dict:
        """
        This function calculates silhouette values for the given dataset, labels, and centroids.

        Args:
            X (pd.DataFrame): The dataset.
            y (pd.DataFrame): The labels of the given dataset.
            centroids (array): The centroids of the given dataset (depending on the K value).

        Returns:
            dict: A dictionary containing silhouette values.
        """
        silhouette_results = {}
        silhouette_results['Silhouette'] = silhouette_score(X=X, labels=y)
        silhouette_results['S. Silhouette'] = simplified_silhouette(X=X,
                                                                    labels=y,
                                                                    centroids=centroids,
                                                                    mode='regular',
                                                                    norm_type='min',
                                                                    regularization='L0',
                                                                    eta=1.0)
        silhouette_results['M.S. Silhouette'] = simplified_silhouette(X=X,
                                                                      labels=y,
                                                                      centroids=centroids,
                                                                      mode='heuristic',
                                                                      norm_type='mean',
                                                                      regularization='L0',
                                                                      eta=1.0)
        return silhouette_results

    @staticmethod
    def arrange_results(results: list) -> list:
        """
        This function sorts the input clustering and silhouette results by the K value and logs the silhouette values.

        Args:
            results (list): A list containing all the clustering and silhouette results.

        Returns:
            list: A sorted list of results by the K value.
        """
        sorted_results = sorted(results, key=lambda x: x['K'])
        for result in sorted_results:
            k = result['K']
            sil_log_str = ''
            for sil_name, sil_value in result['Silhouette'].items():
                sil_log_str += f'({sil_name}) - ({"%.4f" % sil_value}), '
            log_service.log('Info', f'[Clustering Service] : Silhouette values for (K={k}) * {sil_log_str[:-2]}')

        return sorted_results
