import time
import pandas as pd
from array import array
import concurrent.futures
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from ClusteringService.Silhouette import heuristic_silhouette_value, simplified_silhouette


class ClusteringService:
    def __init__(self, log_service):
        self.log_service = log_service

    def execute_clustering_service(self, F: pd.DataFrame, n_features: int):
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

        end = time.time()
        self.log_service.log('Debug', f'[Clustering Service] : Total run time in seconds: [{round(end-start, 3)}]')
        return results

    @staticmethod
    def execute_clustering(F: pd.DataFrame, K: int) -> dict:
        results = {}

        results['k'] = K
        results['kmedoids'] = ClusteringService.run_kmedoids(F, K)
        results['silhouette'] = ClusteringService.calculate_silhouette_value(F, results['kmedoids']['labels'],
                                                                             results['kmedoids']['centroids'])
        # results['single_centroid'] = self.is_new_single_centroid(results['kmedoids']['labels'])
        return results

    @staticmethod
    def run_kmedoids(F: pd.DataFrame, K: int, method: str = 'pam', init: str = 'k-medoids++', random_state: int = 42) -> dict:
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
        kmedoids = KMedoids(n_clusters=K, method=method, init=init, random_state=random_state).fit(F)
        results = {
            'labels': kmedoids.labels_,
            'centroids': kmedoids.cluster_centers_
        }
        return results

    @staticmethod
    def calculate_silhouette_value(X: pd.DataFrame, y: pd.DataFrame, centroids: array, silhouette: bool = False,
                                   heuristic_silhouette: bool = True, simplified_regular_silhouette: bool = False,
                                   simplified_heuristic_silhouette: bool = True,
                                   simplified_improved_silhouette: bool = False) -> dict:

        silhouette_results = {}

        if silhouette:
            silhouette_results['silhouette'] = silhouette_score(X, y)
        if heuristic_silhouette:
            silhouette_results['heuristic_silhouette'] = heuristic_silhouette_value(X, y)
        if simplified_regular_silhouette:
            silhouette_results['simplified_regular_silhouette'] = simplified_silhouette(X, y, centroids, mode='regular')
        if simplified_heuristic_silhouette:
            silhouette_results['simplified_heuristic_silhouette'] = simplified_silhouette(X, y, centroids, mode='heuristic')
        if simplified_improved_silhouette:
            silhouette_results['simplified_improved_silhouette'] = simplified_silhouette(X, y, centroids, mode='improved')

        return silhouette_results

    def arrange_results(self, results: list) -> list:
        sorted_results = sorted(results, key=lambda x: x['k'])
        for result in sorted_results:
            k = result['k']
            sil_log_str = ''
            for sil_name, sil_value in result['silhouette'].items():
                sil_log_str += f'({sil_name}) - ({sil_value}), '
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
