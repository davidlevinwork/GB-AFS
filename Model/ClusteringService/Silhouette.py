import numpy as np
from sklearn.metrics.pairwise import distance_metrics


def simplified_silhouette(X, labels, centroids, mode, norm_type, regularization, eta) -> np.ndarray:
    """
    regular = the final calculation includes the values as is *but* we are changing 1 to 0 (like original sil)
    heuristic = the final calculation ignore the 1 values, so we will have fewer items

    L0 = no regularization
    L1 = 1 - (a(i) / (b(i) * eta))
    L2 = 1 - ((a(i) / b(i)) * eta * (a(i))^2)

    [a] - for each data point, we will calculate & add the distance from itself to its centroid.
    [b] - for each data point, we will calculate the MEAN distance from itself to all other centroids.
    """
    metric = distance_metrics()['euclidean']
    labels_list = [*range(0, labels.shape[0], 1)]

    a, b = [], []
    for x, label in zip(X, labels):
        x_centroid = centroids[label]
        not_x_centroid = [centroid for centroid in centroids if not np.array_equal(centroid, x_centroid)]

        a.append(metric(x.reshape(1, -1), x_centroid.reshape(1, -1)).item(0))

        if norm_type == 'min':
            b.append(min([metric(x.reshape(1, -1), not_x_centroid[j].reshape(1, -1))
                          for j in range(len(not_x_centroid))]).item(0))
        elif norm_type == 'mean':
            b.append(np.mean([metric(x.reshape(1, -1), not_x_centroid[j].reshape(1, -1))
                              for j in range(len(not_x_centroid))]))

    if mode == 'regular':
        sil_values = (np.asarray(b) - np.asarray(a)) / np.maximum(np.asarray(a), np.asarray(b))
        sil_values = [sil if sil != 1.0 else 0.0 for sil in sil_values]
        return np.mean(sil_values)
    if mode == 'heuristic':
        a_non_zero = [a_i for a_i in a if not a_i == 0.0]
        a_zero_indexes = np.where(np.array(a) == 0.0)[0]
        clean_index_list = [i for i in labels_list if i not in a_zero_indexes]
        new_b = [i for j, i in enumerate(b) if j in clean_index_list]

        result = mean_simplified_silhouette_value(a_non_zero, new_b, regularization, eta)
        return result


def mean_simplified_silhouette_value(a: list, b: list, regularization: str, eta: float) -> np.ndarray:
    """Calculate the simplified Silhouette value.

    Parameters
    ----------
    a : list
        For each data point, we will calculate & add the distance from itself to its centroid

    b : list
        For each data point, we will calculate the MEAN distance from itself to all other centroids

    regularization: str
        Type of regularization

    eta: int
        ETA value (for regularization)

    Returns
    -------
    numpy.ndarray
        Silhouette value.
    """
    if regularization == 'L0':
        numerator = np.subtract(np.asarray(b), np.asarray(a))
        denominator = np.maximum(np.asarray(a), np.asarray(b))
        sil_value = np.divide(numerator, denominator)
        return np.mean(sil_value)
    if regularization == 'L1':
        a_regularized = np.multiply(np.asarray(a), eta)
        b_regularized = np.multiply(np.asarray(b), eta)
        numerator = np.subtract(np.asarray(b), np.asarray(a))
        denominator = np.maximum(np.asarray(a_regularized), np.asarray(b_regularized))
        sil_value = np.divide(numerator, denominator)
        return np.mean(sil_value)
    if regularization == 'L2':
        regularization = eta * (np.square(np.asarray(a)))
        numerator = np.subtract(np.asarray(b), np.asarray(a))
        denominator = np.maximum(np.asarray(a), np.asarray(b))
        sil_value = (np.divide(numerator, denominator)) + regularization
        return np.mean(sil_value)
