import os
import time
import numpy as np
import pandas as pd
from array import array
from config import config
import concurrent.futures
from sklearn_extra.cluster import KMedoids
from Model.LogService.Log import log_service
from sklearn.metrics import silhouette_score
from Model.VisualizationService.Visualization import visualization_service
from Model.ClusteringService.Silhouette import optimized_simplified_silhouette


class ClusteringService:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)

    def run(self, feature_matrix: np.ndarray, k_values: list, stage: str, fold_index: int) -> list:
        """
        Execute the clustering service for the given feature matrix and K values.

        Args:
            feature_matrix (np.ndarray): The low-dimensional feature matrix.
            k_values (list): A list of K values to test.
            stage (str): The stage of the algorithm ('Train', 'Full Train', 'Test').
            fold_index (int): The index of the given k-fold (used for saving results & plots).

        Returns:
            list: A list of tuples containing clustering and silhouette results for each K value.
        """
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [executor.submit(self._execute_clustering, feature_matrix, k) for k in k_values]
            results = [task.result() for task in concurrent.futures.as_completed(tasks)]

        results = self._arrange_results(results)

        if config.plots.silhouette:
            visualization_service.plot_silhouette(results, stage, fold_index)
        if config.plots.clustering:
            visualization_service.plot_clustering(feature_matrix, results, stage, fold_index)
        if config.plots.clustering_based_jm:
            visualization_service.plot_clustering_based_jm(feature_matrix, results, stage, fold_index)

        end_time = time.time()
        log_service.log('Debug', f'[Clustering Service] : '
                                 f'Total run time in seconds: [{round(end_time - start_time, 3)}]')
        return results

    @staticmethod
    def _execute_clustering(feature_matrix: pd.DataFrame, k: int) -> dict:
        """
        Execute K-Medoid clustering for the given feature matrix and K value.

        Args:
            feature_matrix (pd.DataFrame): The low-dimensional feature matrix.
            k (int): The number of clusters.

        Returns:
            dict: A dictionary containing K-Medoids and silhouette results for the given K.
        """
        kmedoids_result = ClusteringService._run_kmedoids(feature_matrix, k)
        silhouette_result = ClusteringService._calculate_silhouette_value(
            X=feature_matrix,
            labels=kmedoids_result['Labels'],
            centroids=kmedoids_result['Centroids']
        )

        return {
            'K': k,
            'Kmedoids': kmedoids_result,
            'Silhouette': silhouette_result
        }

    @staticmethod
    def _run_kmedoids(feature_matrix: pd.DataFrame, k: int, init: str = 'k-medoids++') -> dict:
        """
        Perform K-medoids clustering on the given feature matrix

        Args:
            feature_matrix (pd.DataFrame): The reduced feature similarity matrix.
            k (int): The number of clusters.
            init (str, optional): The medoid initialization method ('random', 'heuristic', 'k-medoids++', 'build').
                                  Defaults to 'k-medoids++'.

        Returns:
            dict: A dictionary containing cluster labels and centroids.
        """
        kmedoids = KMedoids(init=init, n_clusters=k, method=config.k_medoids.method).fit(feature_matrix)

        return {
            'Labels': kmedoids.labels_,
            'Features': kmedoids.medoid_indices_,
            'Centroids': kmedoids.cluster_centers_
        }

    @staticmethod
    def _calculate_silhouette_value(X: pd.DataFrame, labels: pd.DataFrame, centroids: array) -> dict:
        """
        Calculate silhouette values for the given dataset, labels, and centroids.

        Args:
            X (pd.DataFrame): The dataset.
            labels (pd.DataFrame): The labels of the given dataset.
            centroids (array): The centroids of the given dataset (depending on the K value).

        Returns:
            dict: A dictionary containing silhouette values.
        """
        silhouette_results = {}

        if config.mode == "full":
            silhouette_results['Silhouette'] = silhouette_score(X=X, labels=labels)
            silhouette_results['SS'] = optimized_simplified_silhouette(
                X=X,
                labels=labels,
                centroids=centroids,
                mode='regular',
                norm_type='min',
                regularization='L0',
                eta=1.0
            )

        silhouette_results['MSS'] = optimized_simplified_silhouette(
            X=X,
            labels=labels,
            centroids=centroids,
            mode='heuristic',
            norm_type='mean',
            regularization='L0',
            eta=1.0
        )

        return silhouette_results

    @staticmethod
    def _arrange_results(results: list) -> list:
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
            silhouette_values = ', '.join(f'({name}) - ({"%.4f" % value})' for name, value in result['Silhouette'].items())
            log_service.log('Info', f'[Clustering Service] : Silhouette values for (K={k}) * {silhouette_values}')

        return sorted_results
