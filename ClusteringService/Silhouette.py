import numpy as np
from sklearn.metrics.pairwise import distance_metrics


def simplified_silhouette(X, labels, centroids, mode='regular', B_type='min', regularization='L0', eta=1.0,
                          metric='euclidean'):
    """
    regular = the final mean calculation includes all the values as is (meaning with value 1) --> #1 = #clustrers
    improved = the final mean calculation includes the values as is *but* we are changing 1 to 0 (like implemented sil)
    heuristic = the final mean calculation ignore the 1 values, so we will have fewer items

    L0 = no regularization
    L1 = 1 - (a(i) / (b(i) * eta))
    L2 = 1 - ((a(i) / b(i)) * eta * (a(i))^2)
    """
    metric = distance_metrics()[metric]
    labels_list = [*range(0, labels.shape[0], 1)]

    a, b = [], []
    for x, label in zip(X, labels):
        x_centroid = centroids[label]
        not_x_centroid = [centroid for centroid in centroids if not np.array_equal(centroid, x_centroid)]

        a.append(metric(x.reshape(1, -1), x_centroid.reshape(1, -1)).item(0))

        if B_type == 'min':
            b.append(min([metric(x.reshape(1, -1), not_x_centroid[j].reshape(1, -1))
                          for j in range(len(not_x_centroid))]).item(0))
        elif B_type == 'mean':
            b.append(np.mean([metric(x.reshape(1, -1), not_x_centroid[j].reshape(1, -1))
                              for j in range(len(not_x_centroid))]))

    if mode == 'regular':
        sil_values = (np.asarray(b) - np.asarray(a)) / np.maximum(np.asarray(a), np.asarray(b))
        return np.mean(sil_values)
    if mode == 'improved':
        sil_values = (np.asarray(b) - np.asarray(a)) / np.maximum(np.asarray(a), np.asarray(b))
        sil_values = [sil if sil != 1.0 else 0.0 for sil in sil_values]
        return np.mean(sil_values)
    if mode == 'heuristic':
        a_non_zero = [a_i for a_i in a if not a_i == 0.0]
        a_zero_indexes = np.where(np.array(a) == 0.0)[0]
        clean_index_list = [i for i in labels_list if i not in a_zero_indexes]
        new_b = [i for j, i in enumerate(b) if j in clean_index_list]

        result = get_simplified_silhouette_value(a_non_zero, new_b, regularization, eta)
        return result


def get_simplified_silhouette_value(a: list, b: list, regularization: str, eta: float):
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


def heuristic_silhouette_value(X, labels, metric='euclidean'):
    """Compute the mean Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster distance (``a``) and the mean nearest-cluster
    distance (``b``) for each sample. The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    To clarify, ``b`` is the distance between a sample and the nearest cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels is ``2 <= n_labels <= n_samples - 1``.

    This function returns the mean Silhouette Coefficient over all samples.

    Parameters
    ----------
    X : A feature array.

    labels: Predicted labels for each sample.

    metric: The metric to use when calculating distance between instances in a feature array.
    """
    metric = distance_metrics()[metric]
    labels_list = [*range(0, labels.shape[0], 1)]

    A = np.array([intra_cluster_distance(X, labels, metric, i) for i in labels_list])

    # Ignore all NaN values
    clean_A = A[np.isfinite(A)]
    A_nan_indexes = np.where(np.isnan(A))[0].tolist()

    relevant_labels_list = [i for i in labels_list if i not in A_nan_indexes]

    B = np.array([nearest_cluster_distance(X, labels, metric, i) for i in relevant_labels_list])

    sil_samples = (B - clean_A) / np.maximum(clean_A, B)
    return np.mean(sil_samples)


def average_silhouette(centroids, metric='euclidean'):
    distances = []
    metric = distance_metrics()[metric]

    for i in range(0, len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(
                metric(centroids[0].reshape(1, -1), centroids[1].reshape(1, -1)).item()
            )
    return sum(distances) / len(distances)


def slow_silhouette(X, labels, metric='euclidean'):
    n_labels = labels.shape[0]
    metric = distance_metrics()[metric]

    A = np.array([intra_cluster_distance(X, labels, metric, i) for i in range(n_labels)])
    B = np.array([nearest_cluster_distance(X, labels, metric, i) for i in range(n_labels)])

    # Ignore all NaN values
    clean_A = A[np.isfinite(A)]
    A_nan_indexes = np.where(np.isnan(A))[0].tolist()

    for index in sorted(A_nan_indexes, reverse=True):
        B = np.delete(B, index)

    sil_samples = (B - clean_A) / np.maximum(clean_A, B)
    return np.mean(sil_samples)


def intra_cluster_distance(X, labels, metric, i):
    indices = np.where(labels == labels[i])[0]
    if len(indices) == 0:
        return 0.
    a = np.mean([metric(X[i].reshape(1, -1), X[j].reshape(1, -1))[0] for j in indices if not i == j])
    return a


def nearest_cluster_distance(X, labels, metric, i):
    label = labels[i]
    b = np.min([
        np.mean([metric(X[i].reshape(1, -1), X[j].reshape(1, -1))[0]
                 for j in np.where(labels == cur_label)[0]])
        for cur_label in set(labels) if not cur_label == label])
    return b
