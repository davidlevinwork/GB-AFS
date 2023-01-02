import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

dt = datetime.now()
time_stamp = datetime.timestamp(dt)
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


class VisualizationService:
    def __init__(self, log_service):
        self.log_service = log_service

    def save_plot(self, file: plt, title: str):
        try:
            current_dir = os.path.dirname(__file__)
            results_dir = os.path.join(current_dir, '../', 'Files/', 'Plots/', f'{time_stamp}/')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            file.savefig(results_dir + title)
        except AssertionError as ex:
            self.log_service.log('Error',
                                 f'[Visualization Service] - Failed to save plot graph as a file. Error: [{ex}]')

    def plot_tsne(self, data, distance_measure: str):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            plt.plot(data[:, 0], data[:, 1], 'o')

            plt.xlabel('First Dimension (x)')
            plt.ylabel('Second Dimension (y)')
            plt.title(f't-SNE Result\n'
                      f'{distance_measure}')

            self.save_plot(plt, f'tSNE Graph - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot t-SNE graph. Error: [{ex}]')

    def plot_silhouette(self, clustering_results: list, distance_measure: str):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            for sil_type in list(clustering_results[0]['silhouette'].keys()):
                k_values = [res['k'] for res in clustering_results]
                sil_values = [res['silhouette'][sil_type] for res in clustering_results]
                plt.plot(k_values, sil_values, label=sil_type, linestyle="solid")

            plt.legend()
            # plt.show()
            self.save_plot(plt, f'Silhouette Graph - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot silhouette graph. Error: [{ex}]')

    def plot_clustering(self, k, labels, centroids, data, distance_measure):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            u_labels = np.unique(labels)

            for label in u_labels:
                plt.scatter(data[labels == label, 0], data[labels == label, 1], label=label)

            plt.scatter(centroids[:, 0], centroids[:, 1], s=80, marker='o', color='black', label="Centroids",
                        facecolors='none')
            plt.xlabel('First Dimension (x)')
            plt.ylabel('Second Dimension (y)')
            plt.title(f'Clustering Result for K={k}\n'
                      f'{distance_measure}')

            plt.legend()
            self.save_plot(plt, f'{k} - Clustering - [{distance_measure}]')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot clustering graph for {k} clusters.'
                                          f' Error: [{ex}]')
