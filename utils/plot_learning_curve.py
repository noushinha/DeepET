import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Smoothing the plots
def smooth_curve(points, factor=0.8):
    """ This function smooths the fluctuation of a plot by
        calculating the moving average of the points based on factor
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_train_vs_vald(csv_content, header):
    """
    This function plots the accuracy/loss of training process versus the validation.
    """
    colors_list = ["black", "crimson", "dodgerblue", "limegreen", "orange", "silver"]
    marker_list = ["s", "s", "o", "o", "v", "v"]
    markers_step = list(range(1, 301, 24))
    epochs = range(1, len(csv_content) + 1)
    fig, ax = plt.subplots()
    for col in range(csv_content.shape[1]):
        points = csv_content[:, col]
        plt.plot(epochs, smooth_curve(points), label=str(header[col]), marker=marker_list[col], color=colors_list[col],
                 markersize=5, markevery=markers_step, linewidth=1.0)
        # plt.setp(lines1, color=colors_list[col], linewidth=1.0)
    ax.set_xlim(1, 301)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Epochs", weight="bold")
    plt.ylabel("Loss", weight="bold")
    plt.legend()

    filename1 = "LearningCurve.eps"
    plt.savefig(filename1, format='eps', dpi=500, bbox_inches="tight")
    filename2 = "LearningCurve.png"
    plt.savefig(filename2, format='png', dpi=500, bbox_inches="tight")
    plt.show()


csv_path = '/mnt/Data/Cryo-ET/DeepET/loss_results.csv'
content = pd.read_csv(csv_path)
header = list(content.columns.values)
content = np.asarray(content.values)
plot_train_vs_vald(content, header)
