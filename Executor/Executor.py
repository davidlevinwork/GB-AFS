from LogService import Log
from DataService import Data
from TableService import Table
from ClusteringService import Clustering
from VisualizationService import Visualization
from ClassificationService import Classification
from FeatureSimilarityService import FeatureSimilarity
from DimensionReductionService import DimensionReduction

from kneed import KneeLocator
from sklearn.model_selection import KFold


class Executor:
    def __init__(self):
        self.log_service = None
        self.data_service = None
        self.table_service = None
        self.clustering_service = None
        self.visualization_service = None
        self.classification_service = None
        self.feature_similarity_service = None
        self.dimension_reduction_service = None

    def init_services(self):
        self.log_service = Log.LogService()
        self.table_service = Table.TableService()
        self.visualization_service = Visualization.VisualizationService(self.log_service)
        self.data_service = Data.DataService(self.log_service, self.visualization_service)
        self.clustering_service = Clustering.ClusteringService(self.log_service, self.visualization_service)
        self.classification_service = Classification.ClassificationService(self.log_service, self.visualization_service)
        self.feature_similarity_service = FeatureSimilarity.FeatureSimilarityService(self.log_service,
                                                                                     self.visualization_service)
        self.dimension_reduction_service = DimensionReduction.DimensionReductionService(self.log_service,
                                                                                        self.visualization_service)

    def execute(self):
        data = self.data_service.execute_data_service(data_set='_Cardiotocography')

        knees = self.execute_train(data=data)
        K_values = [knees['Interp1d']['Knee']] + [knees['Polynomial']['Knee']]
        results = self.stage_2(data=data, K_values=list(set(K_values)), knees=knees)
        self.stage_3(data=data, K_values=list(set(K_values)), knees=knees)

    def execute_train(self, data: dict) -> dict:
        train_results = self.execute_algorithm(data)
        knees = self.find_best_k_value(train_results)
        self.visualization_service.plot_accuracy_to_silhouette(train_results['Classification'],
                                                               train_results['Clustering'], knees,
                                                               'Train')

        return knees

    def stage_2(self, data: dict, K_values: list, knees):
        self.log_service.log('Info', f'[Executor] : ********************* Train *********************')
        F = self.feature_similarity_service.calculate_separation_matrix(X=data['Train'][0], features=data['Features'],
                                                                        labels=data['Labels'],
                                                                        distance_measure='Jeffries-Matusita')
        F_reduced = self.dimension_reduction_service.tsne(F=F, fold_index=1000, perplexity=10.0)

        clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                            K_values=K_values,
                                                                            fold_index=1000)

        classification_res = self.classification_service.classify(train=(data['Train'][1], data['Train'][2]), val=None,
                                                                  F=F_reduced, clustering_res=clustering_res,
                                                                  features=list(data['Features']), K_values=K_values)

        self.table_service.create_table(4444, classification_res)

        return clustering_res

    def stage_3(self, data: dict, K_values: list, knees):
        self.log_service.log('Info', f'[Executor] : ********************* Test *********************')
        F = self.feature_similarity_service.calculate_separation_matrix(X=data['Test'][0], features=data['Features'],
                                                                        labels=data['Labels'],
                                                                        distance_measure='Jeffries-Matusita')
        F_reduced = self.dimension_reduction_service.tsne(F=F, fold_index=1000, perplexity=10.0)

        clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                            K_values=K_values,
                                                                            fold_index=1000)

        classification_res = self.classification_service.classify(train=(data['Test'][1], data['Test'][2]), val=None,
                                                                  F=F_reduced, clustering_res=clustering_res,
                                                                  features=list(data['Features']), K_values=K_values)

        self.table_service.create_table(5555, classification_res)

    def execute_algorithm(self, data: dict) -> dict:
        clustering_results = {}
        classification_results = {}

        # Initialize the KFold object with k splits and shuffle the data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, val_index) in enumerate(kf.split(data['Train'][0])):
            self.log_service.log('Info', f'[Executor] : ******************** Fold Number #{i + 1} ********************')

            # Split the data into train & validation
            train, validation = self.data_service.k_fold_split(data=data['Train'][0],
                                                               train_index=train_index,
                                                               val_index=val_index)

            # Calculate the feature similarity matrix
            F = self.feature_similarity_service.calculate_JM_matrix(X=train[0],
                                                                    features=data['Features'],
                                                                    labels=data['Labels'])

            # Reduce dimensionality & Create a feature graph
            F_reduced = self.dimension_reduction_service.tsne(F=F,
                                                              fold_index=i+1,
                                                              perplexity=10.0)

            # Execute clustering service (K-Medoid + Silhouette)
            K_values = [*range(2, len(data['Features']), 1)]
            clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                                K_values=K_values,
                                                                                fold_index=i+1)

            # Execute classification service
            classification_res = self.classification_service.classify(train=(train[1], train[2]),
                                                                      val=(validation[1], validation[2]),
                                                                      F=F_reduced,
                                                                      clustering_res=clustering_res,
                                                                      features=list(data['Features']),
                                                                      K_values=K_values)

            self.table_service.create_table(i + 1, classification_res)

            clustering_results[i] = clustering_res
            classification_results[i] = classification_res

        train_results = self.data_service.get_train_results(classification_results, clustering_results, i)

        return train_results

    @staticmethod
    def find_best_k_value(train_results: dict) -> dict:
        x = [res['K'] for res in train_results['Clustering']]
        y = [res['Silhouette']['Simplified (mean-L0) Heuristic Silhouette'] for res in train_results['Clustering']]

        kn = KneeLocator(
            x,
            y,
            curve='concave',
            direction='increasing',
            interp_method='interp1d',
        )

        kn_res = {
            'Knee': kn.knee,
            'Knee y': kn.knee_y,
            'Knees': kn.all_knees,
            'Knees y': kn.all_knees_y
        }

        kn_poly = KneeLocator(
            x,
            y,
            curve='concave',
            direction='increasing',
            interp_method='polynomial',
        )

        kn_poly_res = {
            'Knee': kn_poly.knee,
            'Knee y': kn_poly.knee_y,
            'Knees': kn_poly.all_knees,
            'Knees y': kn_poly.all_knees_y
        }

        return {
            'Interp1d': kn_res,
            'Polynomial': kn_poly_res
        }
