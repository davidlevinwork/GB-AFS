import math
import numpy as np
import pandas as pd
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.similarity_based.fisher_score import fisher_score

LIBRARY_METHOD = ['ReliefF', 'Fisher-Score']


def get_distance(metric: str, df: pd.DataFrame, feature: str, label1: str, label2: str) -> float:
    """Calculate the similarity distance between the two classes with reference to the given feature

    Parameters
    ----------
    metric : str
        The distance that measures the similarity of two probability distributions

    df : pandas.DataFrame
        Input data

    feature : str
        The feature with reference to which we want to evaluate the similarity between the two classes

    label1: str
        First class

    label2: str
        Second class

    Returns
    -------
    float
        similarity distance between the two classes with reference to the given feature
    """
    X1, X2 = np.array(df.loc[df['class'] == label1][feature]), np.array(df.loc[df['class'] == label2][feature])

    if metric == 'Bhattacharyya':
        dist = bhattacharyya_distance(X1, X2)
    elif metric == 'Jeffries-Matusita':
        dist = jm_distance(X1, X2)
    elif metric == 'Hellinger':
        dist = hellinger_distance(X1, X2)
    elif metric == 'Wasserstein':
        dist = wasserstein_distance(X1, X2)
    else:
        raise ValueError('Metric not implemented.')

    return dist


def get_library_distance(metric: str, X: pd.DataFrame, Y: pd.DataFrame):
    """Calculate the similarity distance of the given dataset X with reference tou the labels Y

    Parameters
    ----------
    metric : str
        The distance that measures the similarity of two probability distributions

    X : pandas.DataFrame
        Input data

    Y : pandas.DataFrame
        Input class labels

    Returns
    -------
    {numpy array}, shape (n_features,)
        fisher score for each feature
    """
    Y = Y.to_numpy().reshape(Y.shape[0])

    if metric == 'ReliefF':
        return reliefF(X.to_numpy(), Y)
    elif metric == 'Fisher-Score':
        return fisher_score(X.to_numpy(), Y)


def bhattacharyya_distance(p, q):
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 0.00000000001
    std_q = q.std() if q.std() != 0 else 0.00000000001

    var_p, var_q = std_p ** 2, std_q ** 2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + 0.5 * np.log((var_p + var_q) / (2 * (std_p * std_q)))
    return b


def jm_distance(p, q):
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def hellinger_distance(p, q):
    b = bhattacharyya_distance(p, q)
    hellinger = math.sqrt(1 - b)
    return hellinger


def wasserstein_distance(p, q):
    from scipy.stats import wasserstein_distance
    dist = wasserstein_distance(p, q)
    return dist
