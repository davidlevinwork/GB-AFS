from kneed import KneeLocator
from collections import Counter


##################################################
# Auxiliary functions for managing train results #
##################################################

def get_train_results(classification_results: dict, clustering_results: dict, n_folds: int) -> dict:
    """Create dictionary that holds all the train results (classification & clustering)

     Parameters
     ----------
     classification_results : dict
         Classification results of the training stage
     clustering_results : dict
         Clustering results of the training stage
     n_folds : int
         Number of folds used during training

     Returns
     -------
     dict
         Train results (sorted & fixed)
     """
    clustering = get_train_clustering(clustering_results)
    classification = get_train_classification(classification_results, n_folds)

    return {
        'Clustering': clustering,
        'Classification': classification
    }


def get_train_classification(results: dict, n_folds: int) -> dict:
    """Create dictionary that holds all the train classification results

     Parameters
     ----------
     results : list
        Clustering results of the training stage
     n_folds: int
        Number of folds used during training

     Returns
     -------
     dict
         Classification results
     """
    # Init with the results of the first fold
    combined_results = results[0]['Test']['Results By Classifiers']

    # Sum
    for i in range(1, n_folds):
        classifiers = results[i]['Test']['Results By Classifiers']
        for classifier, classifier_results in classifiers.items():
            combined_results[classifier] = dict(Counter(combined_results[classifier]) + Counter(classifier_results))

    # Divide
    for classifier, classifier_results in combined_results.items():
        combined_results[classifier] = [x / n_folds for x in list(combined_results[classifier].values())]

    return combined_results


def get_train_clustering(results: dict) -> list:
    """Create dictionary that holds all the train clustering results

     Parameters
     ----------
     results : list
         Clustering results of the training stage

     Returns
     -------
     dict
         Clustering results
     """
    # Init with the results of the first fold
    combined_results = results[0]

    # Sum
    for i in range(1, len(results)):
        for j, result in enumerate(results[i]):
            k = result['K']
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
    """Find the potential K values to stop the algorithm ("knees")

     Parameters
     ----------
     train_results : dict
         Train results of the training stage

     Returns
     -------
     dict
         Clustering results
     """
    x = [res['K'] for res in train_results['Clustering']]
    y = [res['Silhouette']['M.S. Silhouette'] for res in train_results['Clustering']]

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

    kn_poly = KneeLocator(
        x,
        y,
        curve='concave',
        direction='increasing',
        interp_method='polynomial',
    )

    kn_poly_res = {
        'Knee': kn_poly.knee,
        'Knee y': kn_poly.knee_y,
        'Knees': kn_poly.all_knees,
        'Knees y': kn_poly.all_knees_y
    }

    return {
        'Interp1d': kn_res
        # 'Polynomial': kn_poly_res
    }
