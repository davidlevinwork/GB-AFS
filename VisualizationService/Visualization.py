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

    @staticmethod
    def save_plot(file, title):
        try:
            current_dir = os.path.dirname(__file__)
            results_dir = os.path.join(current_dir, '../', 'Files/', 'Plots/', f'{time_stamp}/')

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            file.savefig(results_dir + title)
        except AssertionError as ex:
            pass
            # LogService.print_to_log('Error', f'Failed to save plot graph as a file. Error: [{ex}]')

    @staticmethod
    def plot_tsne(self, data, distance_measure):
        try:
            plt.clf()
            plt.figure(figsize=(10, 8))

            plt.plot(data[:, 0], data[:, 1], 'o')

            plt.xlabel('First Dimension (x)')
            plt.ylabel('Second Dimension (y)')
            plt.title(f't-SNE Result\n'
                      f'{distance_measure}')

            self.save_plot(plt, f'[t-SNE] - [{distance_measure}]')
        except AssertionError as ex:
            pass
            # LogService.print_to_log('Error', f'Failed to plot tSNE graph. Error: [{ex}]')
