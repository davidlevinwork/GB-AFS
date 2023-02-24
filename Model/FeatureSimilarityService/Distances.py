import numpy as np
import pandas as pd


def get_distance(df: pd.DataFrame, feature: str, label1: str, label2: str) -> float:
    """Calculate the similarity distance between the two classes with reference to the given feature

    Parameters
    ----------
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
    return jm_distance(X1, X2)


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
