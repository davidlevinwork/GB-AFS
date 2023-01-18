from LogService import Log
from DataService import Data
from TableService import Table
from ClusteringService import Clustering
from VisualizationService import Visualization
from ClassificationService import Classification
from FeatureSimilarityService import FeatureSimilarity
from DimensionReductionService import DimensionReduction

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
        data = self.data_service.execute_data_service('Cardiotocography')

        # Initialize the KFold object with 5 splits and shuffle the data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(data['Train'][0]):
            X_train = data['Train'][0].iloc[train_index]
            X_val = data['Train'][0].iloc[val_index]

            F = self.feature_similarity_service.calculate_separation_matrix(X=X_train, features=data['Features'],
                                                                            labels=data['Labels'],
                                                                            distance_measure='Jeffries-Matusita')
            F_reduced = self.dimension_reduction_service.tsne(F=F, perplexity=10.0)

            clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                                n_features=len(data['Features']))

            classification_res = self.classification_service.classify(X=data['Train'][1], y=data['Train'][2],
                                                                      F=F_reduced, clustering_res=clustering_res,
                                                                      features=list(data['Features']), n_values=5)
            self.table_service.create_table(classification_res)
