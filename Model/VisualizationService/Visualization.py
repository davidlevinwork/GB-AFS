import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

dt = datetime.now()
time_stamp = datetime.timestamp(dt)


class VisualizationService:
    def __init__(self, log_service):
        self.log_service = log_service
        self.cmap = plt.get_cmap('nipy_spectral')

    def save_plot(self, file: plt, stage: str, header: str, fold_index: str,  title: str):
        try:
            current_dir = os.path.dirname(__file__)
            if stage == 'Train':
                results_dir = \
                    os.path.join(current_dir, '../', 'Files', 'Plots', f'{time_stamp}', f'{stage}',
                                 f'Fold #{fold_index}', f'{header}')
            else:
                results_dir = \
                    os.path.join(current_dir, '../', 'Files/', 'Plots/', f'{time_stamp}', f'{stage}', f'{header}')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            file.savefig(os.path.join(results_dir, title))
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to save plot as a file. Error: [{ex}]')

    def plot_tsne(self, data: np.ndarray, stage: str, fold_index: int):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            c = data[:, 0] + data[:, 1]
            plt.scatter(x=data[:, 0], y=data[:, 1], marker='o', c=c, cmap='Wistia')

            plt.xlabel(r'$\lambda_1\psi_1$')
            plt.ylabel(r'$\lambda_2\psi_2$')
            plt.colorbar()
            plt.title(f't-SNE Result')

            self.save_plot(plt, stage, 'Dimensionality Reduction', f'{fold_index}', 't-SNE')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot t-SNE graph. Error: [{ex}]')

    def plot_silhouette(self, clustering_results: list, stage: str, fold_index: int):
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))

            colors = ['red', 'green', 'blue']

            for sil_type, color in zip(list(clustering_results[0]['Silhouette'].keys()), colors):
                k_values = [res['K'] for res in clustering_results]
                sil_values = [res['Silhouette'][sil_type] for res in clustering_results]
                ax.plot(k_values, sil_values, label=sil_type, linestyle='--', c=color)

            yticks = []
            for i in range(1, 10):
                yticks.append(i / 10)

            ax.grid(True, linestyle='-.', color='gray')

            # Set x and y axis limits to start from 0
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(k_values))

            # Show only the bottom and left ticks
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # Plot labels
            ax.set_yticks(yticks)
            ax.set_xlabel('K values')
            ax.set_ylabel('Silhouette value')
            ax.set_title('Silhouette Graph')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, shadow=True, fancybox=True)

            plt.tight_layout()
            self.save_plot(plt, stage, 'Silhouette', f'{fold_index}', 'Silhouette Graph')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot silhouette graph. Error: [{ex}]')

    def plot_clustering(self, F: pd.DataFrame, clustering_results: list, stage: str, fold_index: int):
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))

            for clustering_result in clustering_results:
                K = clustering_result['K']
                centroids = clustering_result['Kmedoids']['Centroids']
                labels = clustering_result['Kmedoids']['Labels']
                u_labels = np.unique(clustering_result['Kmedoids']['Labels'])
                for label in u_labels:
                    plt.scatter(F[labels == label, 0], F[labels == label, 1])

                plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='black', facecolors='none', linewidth=2)

                # Plot labels
                plt.xlabel(r'$\lambda_1\psi_1$')
                plt.ylabel(r'$\lambda_2\psi_2$')
                plt.title(f'Clustering Result [K={K}]')

                # Show only the bottom and left ticks
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.tight_layout()
                self.save_plot(plt, stage, 'Clustering', f'{fold_index}', f'Clustering for K={K}')
                plt.close()

        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot clustering graph. Error: [{ex}]')

    def plot_upgrade_clustering(self, F: pd.DataFrame, clustering_results: list, stage: str, fold_index: int):
        try:
            c = F[:, 0] + F[:, 1]
            for clustering_result in clustering_results:
                plt.clf()
                fig, ax = plt.subplots(figsize=(8, 6))

                K = clustering_result['K']
                centroids = clustering_result['Kmedoids']['Centroids']
                plt.scatter(F[:, 0], F[:, 1], c=c, cmap='Wistia')
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='black', facecolors='none', linewidth=1.25, label="Ours")

                plt.xlabel(r'$\lambda_1\psi_1$')
                plt.ylabel(r'$\lambda_2\psi_2$')

                # Show only the bottom and left ticks
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.legend()
                plt.tight_layout()
                self.save_plot(plt, stage, 'U-Clustering', f'{fold_index}', f'Clustering for K={K}')
                plt.close()

        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot clustering graph. Error: [{ex}]')

    def plot_accuracy_to_silhouette(self, classification_res: dict, clustering_res: list, knees: dict, stage: str):
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))

            # Get the colors from the color map
            c_index = 0
            colors = [self.cmap(i) for i in np.linspace(0, 1, len(classification_res) + len(knees) +
                                                        len(clustering_res[0]['Silhouette']))]

            # Left Y axis (accuracy)
            for classifier, classifier_val in classification_res.items():
                x_values = [*range(2, len(classification_res[classifier]) + 2, 1)]
                label = VisualizationService.get_classifier_label(str(classifier))
                ax.plot(x_values, classifier_val, linestyle="-.", label=label, c=colors[c_index])
                c_index += 1

            ax.set_xlabel("K value")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle='-.')
            ax.spines['top'].set_visible(False)
            # ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, shadow=True, fancybox=True)

            # Right Y axis (silhouette)
            ax2 = ax.twinx()
            for sil_type in list(clustering_res[0]['Silhouette'].keys()):
                k_values = [res['K'] for res in clustering_res]
                sil_values = [res['Silhouette'][sil_type] for res in clustering_res]
                ax2.plot(k_values, sil_values, label=sil_type, linestyle="solid", c=colors[c_index])
                c_index += 1

            # get handles and labels for both axes
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            labels = labels1 + labels2

            ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
                      ncol=4, shadow=True, fancybox=True)

            # Knees
            for knee, knee_values in knees.items():
                ax2.axvline(x=knee_values['Knee'], linestyle='dotted', c='red')
                ax2.text(knee_values['Knee'], 0.1, 'KNEE', rotation=90, color='red')

            ax2.set_ylabel("Silhouette Value")
            ax2.spines['top'].set_visible(False)

            self.save_plot(plt, stage, 'Accuracy', 'Final', 'Accuracy-Silhouette')
        except AssertionError as ex:
            self.log_service.log('Error', f'[Visualization Service] - Failed to plot accuracy graph. Error: [{ex}]')

    @staticmethod
    def get_classifier_label(classifier):
        if classifier == "LogisticRegression":
            return "Log. Reg."
        if classifier == "DecisionTreeClassifier":
            return "Dec. Tree"
        if classifier == "KNeighborsClassifier":
            return "KNN"
        if classifier == "RandomForestClassifier":
            return "Ran. Forest"

