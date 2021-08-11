# ============================================================================================
# DeepET - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi
# Team Leader: Daniel Baum
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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


def plot_confusion_matrix(cm, classes,
                          eps_dir, epoch,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        cf_nonnormalized_filename = os.path.join(eps_dir, "/NonNormalized_" + str(epoch) + "_Epochs.eps")
        plt.savefig(cf_nonnormalized_filename, format='eps', dpi=500, bbox_inches="tight")
    else:
        cf_normalized_filename = os.path.join(eps_dir, "/Normalized_" + str(epoch) + "_Epochs.eps")
        plt.savefig(cf_normalized_filename, format='eps', dpi=500, bbox_inches="tight")


def plot_train_vs_vald(train_points, vald_points, eps_dir, epoch, is_loss=False):
    """
    This function plots the accuracy/loss of training process versus the validation.
    """
    plot_label = 'Accuracy'
    if is_loss:
        plot_label = 'Loss'

    epochs = range(1, len(train_points) + 1)

    lines1 = plt.plot(epochs, smooth_curve(train_points), label='Training ' + plot_label)
    plt.setp(lines1, color='red', linewidth=1.0)
    lines2 = plt.plot(epochs, smooth_curve(vald_points), 'b-', label='Validation ' + plot_label)
    plt.setp(lines2, color='black', linewidth=1.0)
    plt.title('Training and Validation ' + plot_label)
    plt.xlabel('Epochs')
    plt.ylabel(plot_label)
    plt.legend()
    filename = os.path.join(eps_dir, "Training_Validation_" + plot_label + "_" + str(
        epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_accuracy(model_history, start_point, eps_dir, epoch):
    """
    if cross-fold validation is used, this function plot accuracy in training
    process versus validation process per fold.
    """
    color_map = ['red', 'black', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'violet', 'pink']

    plt.title('Train Accuracy (T) vs Validation Accuracy (V)')

    pointslen = model_history[0].history['acc']
    pointslen = pointslen[start_point:]
    epochs = range(1, len(pointslen) + 1)

    for i in range(10):
        points1 = model_history[i].history['acc']
        points1 = points1[start_point:]
        lines1_1 = plt.plot(epochs, smooth_curve(points1), label='T Fold ' + str(i+1))
        plt.setp(lines1_1, color=color_map[i], linewidth=1.0)

        points2 = model_history[i].history['val_acc']
        points2 = points2[start_point:]
        lines1_2 = plt.plot(epochs, smooth_curve(points2), label='V Fold ' + str(i+1))
        plt.setp(lines1_2, color=color_map[i], linewidth=1.0, linestyle="dashdot")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=5)
    filename = os.path.join(eps_dir, "Folds_Training_Validation_Accuracy_" + str(
        epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_loss(model_history, start_point, eps_dir, epoch):
    """
    if cross-fold validation is used, this function plot loss in training
    process versus validation process per fold.
    """
    color_map = ['red', 'black', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'violet', 'pink']
    plt.title('Train Loss (T) vs Validation Loss (V)')

    pointslen = model_history[0].history['loss']
    pointslen = pointslen[start_point:]
    epochs = range(1, len(pointslen) + 1)

    for i in range(10):
        points1 = model_history[i].history['loss']
        points1 = points1[start_point:]
        lines1_1 = plt.plot(epochs, smooth_curve(points1), label='T Fold ' + str(i+1))
        plt.setp(lines1_1, color=color_map[i], linewidth=1.0)

        points2 = model_history[i].history['val_loss']
        points2 = points2[start_point:]
        lines1_2 = plt.plot(epochs, smooth_curve(points2), label='V Fold ' + str(i+1))
        plt.setp(lines1_2, color=color_map[i], linewidth=1.0, linestyle="dashdot")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=5)
    filename = os.path.join(eps_dir, "Folds_Training_Validation_Loss_" + str(
        epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


def plot_folds_barchart(t_data, v_data, fold_number, fold_dir, epoch):
    """
    This function plots number of samples selected for each fold as a bar chart.
    """
    y_pos = np.arange(6)
    width = 0.32

    barfig = plt.figure(num=fold_number, figsize=(6, 4), dpi=80)
    ax = plt.subplot(111)

    t_values, t_counts = np.unique(t_data, return_counts=True)
    v_values, v_counts = np.unique(v_data, return_counts=True)
    rects1 = ax.bar(y_pos, t_counts, width, color='SkyBlue', alpha=0.5,)
    rects2 = ax.bar(y_pos+width, v_counts, width, color='IndianRed',  alpha=0.5)

    ax.set_ylabel('# Labels')
    ax.set_xlabel('Categories')
    ax.set_xticks(y_pos + width)
    ax.set_xticklabels(('Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'))
    ax.legend((rects1[0], rects2[0]), ('train', 'validation'))
    # ax.set_title('Distribution of labels in each category')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    filename = os.path.join(fold_dir, "Training_Validation_Fold_" + str(fold_number) + "_Distribution_" +
                            str(epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")
    plt.close(barfig)


def autolabel(ax, rects):
    """
    An auxilary function to generate the axis labels of the
    bar charts
    """
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')


# Compute and plot ROC curve and ROC area for each class
def plot_roc(y_test, y_score, classes_num, eps_dir, epoch):
    """
    This function plots ROC for multi-class classification.
    One ROC plot for each class, also it plots micro and macro ROCs.
    """

    # variable definition
    lw = 1  # plot line width
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculating auc, false positive rate and true positive rate for each class
    for i in range(classes_num):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes_num)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classes_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= classes_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # smoothing the micro roc curve
    micro_poly = np.polyfit(fpr["micro"], tpr["micro"], 5)
    micro_poly_y = np.poly1d(micro_poly)(fpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='magenta', linestyle=':', linewidth=1)

    # smoothing the macro roc curve
    macro_poly = np.polyfit(fpr["macro"], tpr["macro"], 5)
    macro_poly_y = np.poly1d(macro_poly)(fpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='blue', linestyle='-.', linewidth=lw)

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--', linewidth=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for each class')
    plt.legend(loc="lower right")
    filename = os.path.join(eps_dir, "Micro_Macro_Avg_ROC_Curve_" + str(epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


# plotting learning rate
def plot_lr(lr_points, eps_dir, epoch):
    """
    This function plots learning rate versus number of epochs.
    """
    epochs = range(1, len(lr_points) + 1)

    lines1 = plt.plot(epochs, lr_points, label='learning rate')
    plt.setp(lines1, color='black', linewidth=1.0)

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    filename = os.path.join(eps_dir, "Learning_Rate_" + str(epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=500, bbox_inches="tight")


def general_plot(data_points, eps_dir, axis_labels, num_classes, epoch, plot_num):
    legend_names = []
    for lbl in range(0, num_classes):
        legend_names.append('class ' + str(lbl))

    plt.figure(num=plot_num, figsize=(8, 6), dpi=100)
    plt.plot(data_points)
    plt.ylabel(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.legend(legend_names)
    plt.grid()

    filename = os.path.join(eps_dir, axis_labels[0] + "_" + str(epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=500, bbox_inches="tight")

    plt.show()
