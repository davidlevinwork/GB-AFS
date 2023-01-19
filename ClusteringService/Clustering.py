import time
import pandas as pd
from array import array
import concurrent.futures
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from ClusteringService.Silhouette import heuristic_silhouette_value, simplified_silhouette


class ClusteringService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def execute_clustering_service(self, F: pd.DataFrame, n_features: int, fold_index: int):
        start = time.time()
        K_values = [*range(2, n_features, 1)]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit a task for each K value
            tasks = [executor.submit(ClusteringService.execute_clustering, F, K) for K in K_values]

            # Wait for the tasks to complete and store the results
            for task in concurrent.futures.as_completed(tasks):
                results.append(task.result())

        results = self.arrange_results(results)
        self.visualization_service.plot_silhouette(results, 'Jeffries-Matusita', fold_index)
        self.visualization_service.plot_clustering(F, results, 'Jeffries-Matusita', fold_index)

        end = time.time()
        self.log_service.log('Debug', f'[Clustering Service] : Total run time in seconds: [{round(end - start, 3)}]')
        return results

    @staticmethod
    def execute_clustering(F: pd.DataFrame, K: int) -> dict:
        results = {}

        results['K'] = K
        results['Kmedoids'] = ClusteringService.run_kmedoids(F, K)
        results['Silhouette'] = ClusteringService.calculate_silhouette_value(F, results['Kmedoids']['Labels'],
                                                                             results['Kmedoids']['Centroids'])
        # results['single_centroid'] = self.is_new_single_centroid(results['kmedoids']['labels'])
        return results

    @staticmethod
    def run_kmedoids(F: pd.DataFrame, K: int, method: str = 'pam', init: str = 'k-medoids++') -> dict:
        """Perform K-medoids clustering

        Parameters
        ----------
        F : pandas.DataFrame
            Reduced (feature similarity) matrix F
        K : int
            Number of clusters
        method: str, optional
            Which algorithm to use ('alternate' is faster while 'pam' is more accurate)
        init: str, optional
            Specify medoid initialization method ('random', 'heuristic', 'k-medoids++', 'build')
        random_state : int, optional
            Random seed for data shuffling, by default 42

        Returns
        -------
        Dictionary
            Cluster labels, Cluster centroids
        """
        kmedoids = KMedoids(n_clusters=K, method=method, init=init).fit(F)
        results = {
            'Labels': kmedoids.labels_,
            'Centroids': kmedoids.cluster_centers_
        }
        return results

    @staticmethod
    def calculate_silhouette_value(X: pd.DataFrame, y: pd.DataFrame, centroids: array, silhouette: bool = True,
                                   heuristic_silhouette: bool = True, simplified_regular_silhouette: bool = False,
                                   simplified_improved_silhouette: bool = False,
                                   simplified_min_heuristic_silhouette: bool = True,
                                   simplified_mean_heuristic_silhouette: bool = True) -> dict:

        silhouette_results = {}

        if silhouette:
            silhouette_results['Silhouette'] = silhouette_score(X, y)
        if simplified_mean_heuristic_silhouette:
            silhouette_results['Simplified (mean-L0) Heuristic Silhouette'] = simplified_silhouette(X, y, centroids,
                                                                                                    mode='heuristic',
                                                                                                    B_type='mean',
                                                                                                    regularization='L0')
        return silhouette_results

    def arrange_results(self, results: list) -> list:
        sorted_results = sorted(results, key=lambda x: x['K'])
        for result in sorted_results:
            k = result['K']
            sil_log_str = ''
            for sil_name, sil_value in result['Silhouette'].items():
                sil_log_str += f'({sil_name}) - ({"%.4f" % sil_value}), '
            self.log_service.log('Info', f'[Clustering Service] : Silhouette values for (K={k}) * {sil_log_str[:-2]}')

        return sorted_results

    @staticmethod
    def is_new_single_centroid(y: pd.DataFrame, number_of_single_centroids: int) -> bool:
        centroids = sorted(list(dict.fromkeys(y)))
        centroids_occurrences = [y.tolist().count(centroid) for centroid in centroids]
        current_number_of_single_centroids = centroids_occurrences.count(1)
        if current_number_of_single_centroids > number_of_single_centroids:
            return True
        return False
