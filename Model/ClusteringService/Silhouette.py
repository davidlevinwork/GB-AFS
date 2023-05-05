import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def optimized_simplified_silhouette(X, labels, centroids, mode, norm_type, regularization, eta) -> np.ndarray:
    a = euclidean_distances(X, centroids[labels]).diagonal()

    b = np.empty_like(a)
    for idx, centroid in enumerate(centroids):
        not_x_centroid = np.delete(centroids, idx, axis=0)
        distances_to_other_centroids = euclidean_distances(X[labels == idx], not_x_centroid)

        if norm_type == 'min':
            b[labels == idx] = distances_to_other_centroids.min(axis=1)
        elif norm_type == 'mean':
            b[labels == idx] = distances_to_other_centroids.mean(axis=1)

    if mode == 'heuristic':
        mask = a != 0
        a = a[mask]
        b = b[mask]

    numerator = b - a
    denominator = np.maximum(a, b)

    if regularization == 'L1':
        denominator = np.maximum(a * eta, b * eta)
    elif regularization == 'L2':
        regularization_term = eta * (a ** 2)
    else:
        regularization_term = 0

    sil_values = numerator / denominator + regularization_term

    if mode == 'regular':
        sil_values[sil_values == 1] = 0

    return np.mean(sil_values)
