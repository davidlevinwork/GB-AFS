import pandas as pd
from config import config
from random import sample
from kneed import KneeLocator
from collections import Counter
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.similarity_based.fisher_score import fisher_score


##################################################
# Auxiliary functions for managing train results #
##################################################

def compile_train_results(classification_results: dict, clustering_results: dict, n_folds: int) -> dict:
    """
    Create a dictionary that holds all the train results (classification & clustering).

    Args:
        classification_results (dict): Classification results of the training stage.
        clustering_results (dict): Clustering results of the training stage.
        n_folds (int): Number of folds used during training.

    Returns:
        dict: Train results (sorted & fixed).
    """
    classification = {}
    clustering = compile_train_clustering_results(clustering_results)

    if config.mode == "full":
        classification = compile_train_classification_results(classification_results, n_folds)

    return {
        'Clustering': clustering,
        'Classification': classification
    }


def compile_train_classification_results(results: dict, n_folds: int) -> dict:
    """
    Create a dictionary that holds all the train classification results.

    Args:
        results (dict): Classification results of the training stage.
        n_folds (int): Number of folds used during training.

    Returns:
        dict: Combined classification results.
    """
    # Init with the results of the first fold
    combined_results = results[0]['Validation']['Results By Classifiers']

    # Sum
    for i in range(1, n_folds):
        classifiers = results[i]['Validation']['Results By Classifiers']
        for classifier, classifier_results in classifiers.items():
            combined_results[classifier] = dict(Counter(combined_results[classifier]) + Counter(classifier_results))

    # Divide
    for classifier, classifier_results in combined_results.items():
        combined_results[classifier] = [x / n_folds for x in list(combined_results[classifier].values())]

    return combined_results


def compile_train_clustering_results(results: dict) -> list:
    """
    Create a dictionary that holds all the train clustering results.

    Args:
        results (dict): Clustering results of the training stage.

    Returns:
        list: Combined clustering results.
    """
    # Init with the results of the first fold
    combined_results = results[0]

    # Sum
    for i in range(1, len(results)):
        for j, result in enumerate(results[i]):
            sub_results = result['Silhouette']
            for sil_name, sil_value in sub_results.items():
                combined_results[j]['Silhouette'][sil_name] += sil_value

    # Divide
    for result in combined_results:
        sub_results = result['Silhouette']
        for sil_name, sil_value in sub_results.items():
            sub_results[sil_name] /= len(results)

    return combined_results


#############################################################################
# Auxiliary function for getting 'best' K values according to train results #
#############################################################################

def find_knees(train_results: dict) -> dict:
    """
    Find the potential K values to stop the algorithm ("knees").

    Args:
        train_results (dict): Train results of the training stage.

    Returns:
        dict: Potential K values.
    """
    x = [res['K'] for res in train_results['Clustering']]
    y = [res['Silhouette']['MSS'] for res in train_results['Clustering']]

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

    return {
        'Interp1d': kn_res
    }


def select_k_best_features(X: pd.DataFrame, y: pd.DataFrame, k: int, algorithm: str):
    """
    Select k best features from the dataset using the specified algorithm.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        k (int): Number of features to select.
        algorithm (str): Algorithm to use for feature selection.

    Returns:
        pd.DataFrame: Dataframe with selected features.
    """
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
