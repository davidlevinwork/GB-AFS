from DataService import Data
from LogService import Log
from ClusteringService import Clustering
from VisualizationService import Visualization
from FeatureSimilarityService import FeatureSimilarity
from DimensionReductionService import DimensionReduction


class Executor:
    def __init__(self):
        self.log_service = None
        self.data_service = None
        self.clustering_service = None
        self.visualization_service = None
        self.feature_similarity_service = None
        self.dimension_reduction_service = None

    def init_services(self):
        self.log_service = Log.LogService()
        self.data_service = Data.DataService(self.log_service)
        self.clustering_service = Clustering.ClusteringService(self.log_service)
        self.visualization_service = Visualization.VisualizationService(self.log_service)
        self.feature_similarity_service = FeatureSimilarity.FeatureSimilarityService(self.log_service)
        self.dimension_reduction_service = DimensionReduction.DimensionReductionService(self.log_service)

    def execute(self):
        data = self.data_service.execute_data_service('Cardiotocography')

        F = self.feature_similarity_service.calculate_separation_matrix(X=data['train'][0], features=data['features'],
                                                                        labels=data['labels'],
                                                                        distance_measure='Jeffries-Matusita')
        F_reduced = self.dimension_reduction_service.tsne(F=F, perplexity=15.0)

        clustering_res = self.clustering_service.execute_clustering_service(F_reduced, len(data['features']))

        x = 5
