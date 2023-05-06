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
from Model.utils import compile_train_results, find_knees, select_k_best_features, summarize_final_results


class Executor:
    def __init__(self):
        self.data_service = Data.DataService()
        self.table_service = Table.TableService()
        self.clustering_service = Clustering.ClusteringService()
        self.classification_service = Classification.ClassificationService()
        self.feature_similarity_service = FeatureSimilarity.FeatureSimilarityService()
        self.dimension_reduction_service = DimensionReduction.DimensionReductionService()

    def run(self):
        """
        Execute the full workflow of the algorithm, consisting of four stages:
        1. Train stage (first_phase)
        2. Full train stage (second_phase)
        3. Test stage (execute_test) - only in full mode
        4. Benchmarks (execute_benchmarks) - only in full mode
        """
        # Prepare the data
        data = self.data_service.run()
        # STAGE 1 --> Train stage
        K_values = self.first_phase(data=data)
        # STAGE 2 --> Full train stage
        final_features = self.second_phase(data=data, K_values=K_values)
        if config.mode == "full":
            # STAGE 3 --> Test stage
            self.execute_test(data=data, features=final_features)
            # STAGE 4 --> Benchmarks
            self.execute_benchmarks(data=data, k=len(final_features))

        summarize_final_results(data=data, final_features=final_features)

    #####################################################################
    # STAGE 1 - Execute algorithm on train folds ==> get K (knee) value #
    #####################################################################
    def first_phase(self, data: dict) -> list:
        """
        Execute the first phase, which is the train stage. This phase aims to get the K (knee) value.

        Args:
            data (dict): A dictionary containing the dataset information.

        Returns:
            list: A list containing the K (knee) value.
        """

        train_results = self.execute_train(data)
        knees = find_knees(train_results)
        K_values = list([knees['Interp1d']['Knee']])
        visualization_service.plot_accuracy_to_silhouette(classification_res=train_results['Classification'],
                                                          clustering_res=train_results['Clustering'],
                                                          knees=knees,
                                                          stage='Train')
        return K_values

    def execute_train(self, data: dict) -> dict:
        """
        Execute the train stage of the algorithm on the given dataset.

        Args:
            data (dict): A dictionary containing the dataset information.

        Returns:
            dict: A dictionary containing the train results.
        """
        clustering_results = {}
        classification_results = {}

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
            if config.mode == "full":
                classification_results[i] = results['Classification']

        train_results = compile_train_results(classification_results=classification_results,
                                              clustering_results=clustering_results,
                                              n_folds=i)
        return train_results

    def train_procedure(self, data: dict, features: np.ndarray, labels: np.ndarray, K_values: list, stage: str,
                        fold_index: int) -> dict:
        """
        Executes the training procedure of the algorithm.

        Args:
            data (dict): A dictionary containing the dataset information.
            features (np.ndarray): A NumPy array containing the features.
            labels (np.ndarray): A NumPy array containing the labels.
            K_values (list): A list containing the K values.
            stage (str): A string representing the stage of the algorithm.
            fold_index (int): An integer representing the fold index.

        Returns:
            dict: A dictionary containing the results of the clustering and classification procedures.
        """

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

        if config.mode == "basic":
            return {"Clustering": clustering_res}
        else:
            classification_res = self.classification_service.classify(mode=stage,
                                                                      data=fixed_data,
                                                                      feature_matrix=F_reduced,
                                                                      clustering_results=clustering_res,
                                                                      feature_names=list(features),
                                                                      k_values=K_values)

            self.table_service.create_table(fold_index=str(fold_index),
                                            stage=stage,
                                            classification_res=classification_res)

            return {"Clustering": clustering_res,
                    "Classification": classification_res}

    ##################################################################################
    # STAGE 2 - Execute algorithm on full train (only) on K ==> get final K features #
    ##################################################################################
    def second_phase(self, data: dict, K_values: list) -> np.ndarray:
        """
        Execute the second phase, which is the full train stage. This phase aims to get the final K features.

        Args:
            data (dict): A dictionary containing the dataset information.
            K_values (list): A list containing the K values.

        Returns:
            np.ndarray: A NumPy array containing the final K features.
        """
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

    ##############################################################################################
    # STAGE 3 (full mode only) - Execute classification on test set with the K selected features #
    ##############################################################################################
    def execute_test(self, data: dict, features: np.ndarray):
        """
        Execute the test stage, which is only applicable in the full mode. The test stage evaluates the performance
        of the classification with the selected K features.

        Args:
            data (dict): A dictionary containing the dataset information.
            features (np.ndarray): A NumPy array containing the selected K features.
        """
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

    ###########################################################################################################
    # STAGE 4 (full mode only) - Execute benchmarks  on test set (each method will define the best K features #
    ###########################################################################################################
    def execute_benchmarks(self, data: dict, k: int):
        """
        Execute the benchmarks stage, which is only applicable in the full mode. This stage evaluates the performance
        of various feature selection methods and compares them against the algorithm's performance.

        Args:
            data (dict): A dictionary containing the dataset information.
            k (int): An integer representing the number of selected features.
        """
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
