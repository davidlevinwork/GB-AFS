from LogService import Log
from DataService import Data
from TableService import Table
from ClusteringService import Clustering
from VisualizationService import Visualization
from ClassificationService import Classification
from FeatureSimilarityService import FeatureSimilarity
from DimensionReductionService import DimensionReduction

from sklearn.model_selection import KFold
from utils import get_train_results, find_knees


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
        # Prepare the data
        data = self.data_service.execute_data_service(data_set='_Cardiotocography')
        # STAGE 1 --> Train stage
        K_values = self.execute_train(data=data)
        # STAGE 2 --> Full train stage
        self.execute_semi_algorithm(data=data, K_values=K_values, mode="Train", log_mode="Full Train")
        # STAGE 3 --> Test stage
        self.execute_semi_algorithm(data=data, K_values=K_values, mode="Test", log_mode="Test")

    def execute_train(self, data: dict) -> list:
        train_results = self.execute_full_algorithm(data)
        knees = find_knees(train_results)
        K_values = list(set([knees['Interp1d']['Knee']] + [knees['Polynomial']['Knee']]))
        self.visualization_service.plot_accuracy_to_silhouette(classification_results=train_results['Classification'],
                                                               clustering_results=train_results['Clustering'],
                                                               knees=knees,
                                                               mode='Train')
        return K_values

    def execute_full_algorithm(self, data: dict) -> dict:
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
            fixed_data = {
                "Train": (train[1], train[2]),
                "Validation": (validation[1], validation[2])
            }
            classification_res = self.classification_service.classify(mode="Train",
                                                                      data=fixed_data,
                                                                      F=F_reduced,
                                                                      clustering_res=clustering_res,
                                                                      features=list(data['Features']),
                                                                      K_values=K_values)
            # Create results table
            self.table_service.create_table(index=f"{i+1}",
                                            classification_res=classification_res)

            clustering_results[i] = clustering_res
            classification_results[i] = classification_res

        train_results = get_train_results(classification_results=classification_results,
                                          clustering_results=clustering_results,
                                          n_folds=i)

        return train_results

    def execute_semi_algorithm(self, data: dict, K_values: list, mode: str, log_mode: str):
        self.log_service.log('Info', f'[Executor] : ********************* {mode} *********************')

        # Calculate the feature similarity matrix
        F = self.feature_similarity_service.calculate_JM_matrix(X=data[f'{mode}'][0],
                                                                features=data['Features'],
                                                                labels=data['Labels'])

        # Reduce dimensionality & Create a feature graph
        F_reduced = self.dimension_reduction_service.tsne(F=F,
                                                          fold_index=1000,
                                                          perplexity=10.0)

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                            K_values=K_values,
                                                                            fold_index=1000)
        # Execute classification service
        fixed_data = {
            f'{mode}': (data[f'{mode}'][1], data[f'{mode}'][2])
        }
        classification_res = self.classification_service.classify(mode=f'{log_mode}',
                                                                  data=fixed_data,
                                                                  F=F_reduced,
                                                                  clustering_res=clustering_res,
                                                                  features=list(data['Features']),
                                                                  K_values=K_values)
        # Create results table
        self.table_service.create_table(f'{log_mode}', classification_res)
