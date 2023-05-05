import numpy as np
from config import config
from Model.DataService import Data
from Model.TableService import Table
from sklearn.model_selection import KFold
from Model.LogService.Log import log_service
from Model.ClusteringService import Clustering
from Model.ClassificationService import Classification
from Model.FeatureSimilarityService import FeatureSimilarity
from Model.DimensionReductionService import DimensionReduction
from Model.VisualizationService.Visualization import visualization_service
from Model.utils import get_train_results, find_knees, select_k_best_features


class Executor:
    def __init__(self):
        self.data_service = Data.DataService()
        self.table_service = Table.TableService()
        self.clustering_service = Clustering.ClusteringService()
        self.classification_service = Classification.ClassificationService()
        self.feature_similarity_service = FeatureSimilarity.FeatureSimilarityService()
        self.dimension_reduction_service = DimensionReduction.DimensionReductionService()

    def run(self):
        # Prepare the data
        data = self.data_service.run()
        # STAGE 1 --> Train stage
        K_values = self.first_phase(data=data)
        # STAGE 2 --> Full train stage
        final_features = self.second_phase(data=data, K_values=K_values)
        # STAGE 3 --> Test stage
        self.execute_test(data=data, features=final_features)
        # STAGE 4 --> Benchmarks
        self.execute_benchmarks(data=data, k=len(final_features))

    ##############################################################
    # STAGE 1 - Execute algorithm on train folds ==> get K value #
    ##############################################################
    def first_phase(self, data: dict) -> list:
        train_results = self.execute_train(data)
        knees = find_knees(train_results)
        K_values = list([knees['Interp1d']['Knee']])
        visualization_service.plot_accuracy_to_silhouette(classification_res=train_results['Classification'],
                                                          clustering_res=train_results['Clustering'],
                                                          knees=knees,
                                                          stage='Train')
        return K_values

    def train_procedure(self, data: dict, features: np.ndarray, labels: np.ndarray, K_values: list, stage: str,
                        fold_index: int) -> dict:
        # Calculate the feature similarity matrix
        F = self.feature_similarity_service.calculate_JM_matrix(X=data['Train'][0],
                                                                features=features,
                                                                labels=labels)

        # Reduce dimensionality & Create a feature graph
        F_reduced = self.dimension_reduction_service.tsne(F=F,
                                                          stage=stage,
                                                          fold_index=fold_index)

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                            stage=stage,
                                                                            K_values=K_values,
                                                                            fold_index=fold_index)

        if "Validation" in data:
            fixed_data = {
                "Train": (data['Train'][1], data['Train'][2]),
                "Validation": (data['Validation'][1], data['Validation'][2])
            }
        else:
            fixed_data = {
                "Train": (data['Train'][1], data['Train'][2])
            }

        classification_res = self.classification_service.classify(mode=stage,
                                                                  data=fixed_data,
                                                                  F=F_reduced,
                                                                  clustering_res=clustering_res,
                                                                  features=list(features),
                                                                  K_values=K_values)

        # Create results table
        self.table_service.create_table(fold_index=str(fold_index),
                                        stage=stage,
                                        classification_res=classification_res)

        return {
            "Clustering": clustering_res,
            "Classification": classification_res
        }

    def execute_train(self, data: dict) -> dict:
        clustering_results = {}
        classification_results = {}

        # Initialize the KFold object with k splits and shuffle the data
        kf = KFold(n_splits=config.k_fold.n_splits,
                   shuffle=config.k_fold.shuffle,
                   random_state=42)

        for i, (train_index, val_index) in enumerate(kf.split(data['Train'][0])):
            log_service.log('Info', f'[Executor] : ******************** Fold Number #{i + 1} ********************')

            train, validation = self.data_service.k_fold_split(data=data['Train'][0],
                                                               train_index=train_index,
                                                               val_index=val_index)
            n_data = {
                "Train": train,
                "Validation": validation
            }

            results = self.train_procedure(data=n_data,
                                           features=data['Features'],
                                           labels=data['Labels'],
                                           K_values=[*range(2, len(data['Features']), 1)],
                                           stage="Train",
                                           fold_index=i + 1)

            clustering_results[i] = results['Clustering']
            classification_results[i] = results['Classification']

        train_results = get_train_results(classification_results=classification_results,
                                          clustering_results=clustering_results,
                                          n_folds=i)

        return train_results

    ############################################################################
    # STAGE 2 - Execute algorithm on full train (only) on K ==> get K features #
    ############################################################################
    def second_phase(self, data: dict, K_values: list) -> np.ndarray:
        log_service.log('Info', f'[Executor] : ********************* Full Train *********************')

        n_data = {
            "Train": data['Train']
        }

        results = self.train_procedure(data=n_data,
                                       features=data['Features'],
                                       labels=data['Labels'],
                                       K_values=K_values,
                                       stage="Full Train",
                                       fold_index=0)

        return results['Clustering'][0]['Kmedoids']['Features']

    #####################################################################
    # STAGE 3 - Execute algorithm on full test (only) on the K features #
    #####################################################################
    def execute_test(self, data: dict, features: np.ndarray):
        log_service.log('Info', f'[Executor] : ********************* Test *********************')

        # Execute classification service
        X, y = data['Test'][1], data['Test'][2]
        new_X = X.iloc[:, features]
        classification_res = self.classification_service.evaluate(X=new_X,
                                                                  y=y,
                                                                  mode="Test",
                                                                  K=len(features))
        final_results = self.classification_service.arrange_results([classification_res])

        # Create results table
        self.table_service.create_table(fold_index="0",
                                        stage="Test",
                                        classification_res={"Test": final_results})

    ################################################################
    # STAGE 4 - Execute benchmarks algorithm on the chosen K value #
    ################################################################
    def execute_benchmarks(self, data: dict, k: int):
        log_service.log('Info', f'[Executor] : ********************* Bench Marks *********************')

        y = data['Test'][2]
        classifications_res = []
        bench_algorithms = ["RELIEF", "FISHER-SCORE", "CFS", "MRMR", "RANDOM"]

        for algo in bench_algorithms:
            log_service.log('Info', f'[Executor] : Executing benchmark with [{algo}] on [{k}].')
            new_X = select_k_best_features(k=k,
                                           algorithm=algo,
                                           X=data['Test'][1],
                                           y=data['Test'][2])
            classification_res = self.classification_service.evaluate(X=new_X,
                                                                      y=y,
                                                                      mode="Test",
                                                                      K=k)
            classifications_res.append(classification_res)

        final_results = self.classification_service.arrange_results(classifications_res)

        # Create results table
        self.table_service.create_table(fold_index="0",
                                        stage="Benchmarks",
                                        classification_res={"Test": final_results})
