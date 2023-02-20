import time
import pandas as pd
from array import array
import concurrent.futures
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from ClusteringService.Silhouette import simplified_silhouette


class ClusteringService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def execute_clustering_service(self, F: pd.DataFrame, K_values: list, stage: str, fold_index: int) -> list:
        """Get all the possible combinations of the given labels

        Parameters
        ----------
        F : pandas.DataFrame
            JM matrix (low dimension)

        K_values: list
            K values that we want to test

        stage: str
            Stage of the algorithm (Train, Full Train, Test)

        fold_index: int
            Index of the given k-fold (for saving results & plots)

        Returns
        -------
        list
            list of Clustering & Silhouette results.
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
        self.visualization_service.plot_silhouette(results, stage, fold_index)
        self.visualization_service.plot_clustering(F, results, stage, fold_index)
        self.visualization_service.plot_upgrade_clustering(F, results, stage, fold_index)

        end = time.time()
        self.log_service.log('Debug', f'[Clustering Service] : Total run time in seconds: [{round(end - start, 3)}]')
        return results

    @staticmethod
    def execute_clustering(F: pd.DataFrame, K: int) -> dict:
        """Execute the K-Medoid clustering algorithm

        Parameters
        ----------
        F : pandas.DataFrame
            JM matrix (low dimension)

        K: int
            Indicates the number of clusters

        Returns
        -------
        dict
            K-Medoids & Silhouette results for the given K.
        """
        results = {}

        results['K'] = K
        results['Kmedoids'] = ClusteringService.run_kmedoids(F, K)
        results['Silhouette'] = ClusteringService.calculate_silhouette_value(X=F,
                                                                             y=results['Kmedoids']['Labels'],
                                                                             centroids=results['Kmedoids']['Centroids'])
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
    def calculate_silhouette_value(X: pd.DataFrame, y: pd.DataFrame, centroids: array) -> dict:
        """Calculate Silhouette values

        Parameters
        ----------
        X : pandas.DataFrame
            Dataset
        y : pandas.DataFrame
            Labels of the given dataset
        centroids: array
            The centroids of the given dataset (depending on K value)

        Returns
        -------
        Dictionary
            Silhouette values.
        """
        silhouette_results = {}
        silhouette_results['Silhouette'] = silhouette_score(X=X, labels=y)
        silhouette_results['Mean Simplified Silhouette'] = simplified_silhouette(X=X,
                                                                                 labels=y,
                                                                                 centroids=centroids,
                                                                                 mode='heuristic',
                                                                                 norm_type='mean',
                                                                                 regularization='L0',
                                                                                 eta=1.0)
        return silhouette_results

    def arrange_results(self, results: list) -> list:
        """Calculate Silhouette values

        Parameters
        ----------
        results : list
            All the results (clustering & Silhouette)

        Returns
        -------
        list
            Sorted results list
        """
        sorted_results = sorted(results, key=lambda x: x['K'])
        for result in sorted_results:
            k = result['K']
            sil_log_str = ''
            for sil_name, sil_value in result['Silhouette'].items():
                sil_log_str += f'({sil_name}) - ({"%.4f" % sil_value}), '
            self.log_service.log('Info', f'[Clustering Service] : Silhouette values for (K={k}) * {sil_log_str[:-2]}')

        return sorted_results
