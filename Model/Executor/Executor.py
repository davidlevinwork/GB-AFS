from Model.LogService import Log
from Model.DataService import Data
from Model.TableService import Table
from Model.ClusteringService import Clustering
from Model.VisualizationService import Visualization
from Model.ClassificationService import Classification
from Model.FeatureSimilarityService import FeatureSimilarity
from Model.DimensionReductionService import DimensionReduction

import pandas
import numpy as np
from random import sample
from sklearn.model_selection import KFold
from Model.utils import get_train_results, find_knees
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.similarity_based.fisher_score import fisher_score


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
        data = self.data_service.execute_data_service(data_set='Microsoft_Malware_Sample')
        # STAGE 1 --> Train stage
        K_values = self.execute_train(data=data)
        # STAGE 2 --> Full train stage
        final_features = self.execute_full_train(data=data, K_values=K_values)
        # STAGE 3 --> Test stage
        self.execute_test(data=data, features=final_features)
        # Stage 4 --> Benchmarks
        self.execute_benchmarks(data=data, k=len(final_features))

    ##############################################################
    # STAGE 1 - Execute algorithm on train folds ==> get K value #
    ##############################################################
    def execute_train(self, data: dict) -> list:
        train_results = self.execute_algorithm(data)
        knees = find_knees(train_results)
        # K_values = list(set([knees['Interp1d']['Knee']] + [knees['Polynomial']['Knee']]))
        K_values = list([knees['Interp1d']['Knee']])
        self.visualization_service.plot_accuracy_to_silhouette(classification_res=train_results['Classification'],
                                                               clustering_res=train_results['Clustering'],
                                                               knees=knees,
                                                               stage='Train')
        return K_values

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
                                                              stage="Train",
                                                              fold_index=i+1,
                                                              perplexity=10.0)

            # Execute clustering service (K-Medoid + Silhouette)
            K_values = [*range(2, len(data['Features']), 1)]
            clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                                stage="Train",
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
            self.table_service.create_table(fold_index=f"{i+1}",
                                            stage="Train",
                                            classification_res=classification_res)

            clustering_results[i] = clustering_res
            classification_results[i] = classification_res

        train_results = get_train_results(classification_results=classification_results,
                                          clustering_results=clustering_results,
                                          n_folds=i)

        return train_results

    ############################################################################
    # STAGE 2 - Execute algorithm on full train (only) on K ==> get K features #
    ############################################################################
    def execute_full_train(self, data: dict, K_values: list) -> np.ndarray:
        self.log_service.log('Info', f'[Executor] : ********************* Train *********************')

        # Calculate the feature similarity matrix
        F = self.feature_similarity_service.calculate_JM_matrix(X=data['Train'][0],
                                                                features=data['Features'],
                                                                labels=data['Labels'])

        # Reduce dimensionality & Create a feature graph
        F_reduced = self.dimension_reduction_service.tsne(F=F,
                                                          stage="Full Train",
                                                          fold_index=0,
                                                          perplexity=10.0)

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_res = self.clustering_service.execute_clustering_service(F=F_reduced,
                                                                            stage="Full Train",
                                                                            K_values=K_values,
                                                                            fold_index=0)

        # Execute classification service
        fixed_data = {
            'Train': (data['Train'][1], data['Train'][2])
        }
        classification_res = self.classification_service.classify(mode="Full Train",
                                                                  data=fixed_data,
                                                                  F=F_reduced,
                                                                  clustering_res=clustering_res,
                                                                  features=list(data['Features']),
                                                                  K_values=K_values)

        # Create results table
        self.table_service.create_table(fold_index="0",
                                        stage="Full Train",
                                        classification_res=classification_res)

        return clustering_res[0]['Kmedoids']['Features']

    #####################################################################
    # STAGE 3 - Execute algorithm on full test (only) on the K features #
    #####################################################################
    def execute_test(self, data: dict, features: np.ndarray):
        self.log_service.log('Info', f'[Executor] : ********************* Test *********************')

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
    def execute_benchmarks(self, data: pandas.DataFrame, k: int):
        self.log_service.log('Info', f'[Executor] : ********************* Bench Marks *********************')

        y = data['Test'][2]
        classifications_res = []
        bench_algorithms = ["RELIEF", "FISHER-SCORE", "CFS", "MRMR", "RANDOM"]

        for algo in bench_algorithms:
            self.log_service.log('Info', f'[Executor] : Executing benchmark with [{algo}] on [{k}].')
            new_X = Executor.select_k_best_features(k=k,
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

    @staticmethod
    def select_k_best_features(X: pandas.DataFrame, y: pandas.DataFrame, k: int, algorithm: str):
        y = y.to_numpy().reshape(y.shape[0])

        if algorithm == "RELIEF":
            score = reliefF(X.to_numpy(), y)
            selected_features = X.columns[score.argsort()[-k:]].tolist()
        elif algorithm == "FISHER-SCORE":
            score = fisher_score(X.to_numpy(), y)
            selected_features = X.columns[score.argsort()[-k:]].tolist()
        elif algorithm == "CFS":
            score = cfs(X.to_numpy(), y)
            selected_features = X.columns[score.argsort()[-k:]].tolist()
        elif algorithm == "MRMR":
            score = mrmr(X.to_numpy(), y, k)
            selected_features = X.columns[score].tolist()
        elif algorithm == "RANDOM":
            selected_features = sample(X.columns.tolist(), k)
        else:
            raise ValueError("Invalid algorithm name")

        return X[selected_features]
