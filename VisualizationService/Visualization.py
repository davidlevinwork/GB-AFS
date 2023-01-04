import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

dt = datetime.now()
time_stamp = datetime.timestamp(dt)


class VisualizationService:
    def __init__(self, log_service):
        self.log_service = log_service
        self.cmap = plt.get_cmap('gist_ncar')

    def save_plot(self, file: plt, header: str, title: str):
        try:
            current_dir = os.path.dirname(__file__)
            results_dir = os.path.join(current_dir, '../', 'Files/', 'Plots/', f'{time_stamp}', f'{header}')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            file.savefig(os.path.join(results_dir, title))
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to save plot as a file. Error: [{ex}]')

    def plot_tsne(self, data: np.ndarray, distance_measure: str):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            plt.plot(data[:, 0], data[:, 1], 'o')

            plt.xlabel('First Dimension (x)')
            plt.ylabel('Second Dimension (y)')
            plt.title(f't-SNE Result\n'
                      f'{distance_measure}')

            self.save_plot(plt, 'Dimensionality Reduction', f'tSNE Graph - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot t-SNE graph. Error: [{ex}]')

    def plot_silhouette(self, clustering_results: list, distance_measure: str):
        try:
            plt.clf()
            plt.figure(figsize=(12, 10))
            color = iter(cm.rainbow(np.linspace(0, 1, len(clustering_results[0]['Silhouette'].keys()))))

            # Get the colors from the color map
            c_index = 0
            colors = [self.cmap(i) for i in np.linspace(0, 1, len(clustering_results[0]['Silhouette'].keys()))]

            for sil_type in list(clustering_results[0]['Silhouette'].keys()):
                k_values = [res['K'] for res in clustering_results]
                sil_values = [res['Silhouette'][sil_type] for res in clustering_results]
                plt.plot(k_values, sil_values, label=sil_type, linestyle="solid", c=colors[c_index])
                c_index += 1

            plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.)
            self.save_plot(plt, 'Silhouette', f'Silhouette Graph - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot silhouette graph. Error: [{ex}]')

    def plot_clustering(self, F: pd.DataFrame, clustering_results: list, distance_measure: str):
        try:
            plt.clf()
            plt.figure(figsize=(12, 10))

            for clustering_result in clustering_results:
                K = clustering_result['K']
                centroids = clustering_result['Kmedoids']['Centroids']
                labels = clustering_result['Kmedoids']['Labels']
                u_labels = np.unique(clustering_result['Kmedoids']['Labels'])
                for label in u_labels:
                    plt.scatter(F[labels == label, 0], F[labels == label, 1])

                plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='black', facecolors='none')

                plt.xlabel('First Dimension (x)')
                plt.ylabel('Second Dimension (y)')
                plt.title(f'Clustering Result for K={K}\n'
                          f'{distance_measure}')

                self.save_plot(plt, 'Clustering', f'Clustering for K={K} - [{distance_measure}]')

        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot clustering graph. Error: [{ex}]')

    def plot_accuracy_to_silhouette(self, classification_results: dict, clustering_results: list, distance_measure: str,
                                    mode: str, sil_min: int = 0.0, sil_max: int = 1.0, acc_min: int = 0.0,
                                    acc_max: int = 1.0):
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 10))
            color = iter(cm.rainbow(np.linspace(0, 1, (len(classification_results['Results By Classifiers'].keys()) +
                                                       len(clustering_results[0]['Silhouette'].keys())))))

            # Get the colors from the color map
            c_index = 0
            colors = [self.cmap(i) for i in np.linspace(0, 1,
                                                        len(classification_results['Results By Classifiers'].keys()) +
                                                        len(clustering_results[0]['Silhouette'].keys()))]

            # Left Y axis (accuracy)
            for classifier, classifier_val in classification_results['Results By Classifiers'].items():
                ax.plot(classifier_val.keys(), classifier_val.values(), linestyle="-.", label=f'{str(classifier)}',
                        c=colors[c_index])
                c_index += 1

            ax.set_xlabel("K value")
            ax.set_ylabel("Accuracy")
            ax.axis(ymin=acc_min, ymax=acc_max)
            ax.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.)

            # Right Y axis (silhouette)
            ax2 = ax.twinx()
            for sil_type in list(clustering_results[0]['Silhouette'].keys()):
                k_values = [res['K'] for res in clustering_results]
                sil_values = [res['Silhouette'][sil_type] for res in clustering_results]
                plt.plot(k_values, sil_values, label=sil_type, linestyle="solid", c=colors[c_index])
                c_index += 1

            ax2.set_ylabel("Silhouette Value")
            ax2.axis(ymin=sil_min, ymax=sil_max)
            ax2.legend(bbox_to_anchor=(0.01, 1), loc=2, borderaxespad=0.)

            self.save_plot(plt, 'Accuracy', f'Accuracy-Silhouette {mode} - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot accuracy graph. Error: [{ex}]')
