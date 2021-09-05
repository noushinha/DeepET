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
import seaborn as sns
from itertools import cycle, product
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.ticker import MaxNLocator
from utils import params
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
                          eps_dir,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not normalize:
        cf_nonnormalized_filename1 = os.path.join(eps_dir, "evaluate/NonNormalized_ConfMtrx" + ".eps")
        cf_nonnormalized_filename2 = os.path.join(eps_dir, "evaluate/NonNormalized_ConfMtrx" + ".png")
        plt.savefig(cf_nonnormalized_filename1, format='eps', dpi=300, bbox_inches="tight")
        plt.savefig(cf_nonnormalized_filename2, format='png', dpi=300, bbox_inches="tight")
    else:
        cf_normalized_filename1 = os.path.join(eps_dir, "evaluate/Normalized_ConfMtrx" + ".eps")
        cf_normalized_filename2 = os.path.join(eps_dir, "evaluate/Normalized_ConfMtrx" + ".png")
        plt.savefig(cf_normalized_filename1, format='eps', dpi=300, bbox_inches="tight")
        plt.savefig(cf_normalized_filename2, format='png', dpi=300, bbox_inches="tight")


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
def plot_roc(y_test, y_score, classes_num, eps_dir):
    """
    This function plots ROC for multi-class classification.
    One ROC plot for each class, also it plots micro and macro ROCs.
    """

    # variable definition
    lw = 1  # plot line width
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    # y_test = y_test[1:y_test.shape[0], 1:y_test.shape[1]]
    # y_score = y_test[1:y_score.shape[0], 1:y_score.shape[1]]
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
    # micro_poly = np.polyfit(fpr["micro"], tpr["micro"], 5)
    # micro_poly_y = np.poly1d(micro_poly)(fpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro ROC, (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='navy', linestyle=':', linewidth=1)

    # smoothing the macro roc curve
    # macro_poly = np.polyfit(fpr["macro"], tpr["macro"], 5)
    # macro_poly_y = np.poly1d(macro_poly)(fpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro ROC, (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle='-.', linewidth=lw)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    palette = cycle(sns.color_palette())

    for i, color in zip(range(classes_num), palette):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {cls} ROC, (AUC = {arr})'
                       ''.format(cls=params.class_names[i], arr=np.round(roc_auc[i],2)))

    plt.plot([0, 1], [0, 1], color='silver', linestyle='--', linewidth=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for each class')
    plt.legend(loc="best")
    filename1 = os.path.join(eps_dir, "evaluate/Micro_Macro_Avg_ROC_Curve.eps")
    filename2 = os.path.join(eps_dir, "evaluate/Micro_Macro_Avg_ROC_Curve.png")
    plt.savefig(filename1, format='eps', dpi=300, bbox_inches="tight")
    plt.savefig(filename2, format='png', dpi=300, bbox_inches="tight")


def plot_recall_precision(y_test, y_score, classes_num, eps_dir):
    """
    This function plots precision Vs. Recall for multi-class classification.
    One precision recall plot for each class, also it plots micro and macro ROCs.
    """

    # variable definition
    lw=1
    lr_recall = dict()
    lr_precision = dict()
    rp_auc = dict()

    # calculating auc, recall and precision for each class
    for i in range(classes_num):
        lr_precision[i], lr_recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        rp_auc[i] = auc(lr_recall[i], lr_precision[i])


    # Compute micro-average ROC curve and ROC area
    lr_precision["micro"], lr_recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    rp_auc["micro"] = auc(lr_recall["micro"], lr_precision["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_recalls = np.unique(np.concatenate([lr_recall[i] for i in range(classes_num)]))

    # Then interpolate all ROC curves at this points
    mean_precisions = np.zeros_like(all_recalls)
    for i in range(classes_num):
        mean_precisions += np.interp(all_recalls, lr_recall[i], lr_precision[i])

    # Finally average it and compute AUC
    mean_precisions /= classes_num

    lr_recall["macro"] = all_recalls
    lr_precision["macro"] = mean_precisions
    rp_auc["macro"] = auc(lr_recall["macro"], lr_precision["macro"])

    # smoothing the micro roc curve
    # micro_poly = np.polyfit(fpr["micro"], tpr["micro"], 5)
    # micro_poly_y = np.poly1d(micro_poly)(fpr["micro"])
    plt.plot(lr_recall["micro"], lr_precision["micro"],
             label='micro, (AUC = {0:0.2f})'
                   ''.format(rp_auc["micro"]),
             color='navy', linestyle=':', linewidth=1)

    # smoothing the macro roc curve
    # macro_poly = np.polyfit(fpr["macro"], tpr["macro"], 5)
    # macro_poly_y = np.poly1d(macro_poly)(fpr["macro"])
    plt.plot(lr_recall["macro"], lr_precision["macro"],
             label='macro, (AUC = {0:0.2f})'
                   ''.format(rp_auc["macro"]),
             color='black', linestyle='-.', linewidth=lw)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    palette = cycle(sns.color_palette())

    for i, color in zip(range(classes_num), palette):
        plt.plot(lr_recall[i], lr_precision[i], color=color, lw=lw,
                 label='class {cls}, (AUC = {arr})'
                       ''.format(cls=params.class_names[i], arr=np.round(rp_auc[i], 2)))

    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Class-based Recall Precision Curve ')
    plt.legend(loc="best")
    filename1 = os.path.join(eps_dir, "evaluate/Recall_Precision_Curve.eps")
    filename2 = os.path.join(eps_dir, "evaluate/Recall_Precision_Curve.png")
    plt.savefig(filename1, format='eps', dpi=300, bbox_inches="tight")
    plt.savefig(filename2, format='png', dpi=300, bbox_inches="tight")

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
    plt.savefig(filename, format='eps', dpi=300, bbox_inches="tight")


def general_plot(data_points, eps_dir, axis_labels, class_names, epoch, plot_num):
    legend_names = []
    class_names = class_names.split(",")
    for lbl in range(len(class_names)):
        legend_names.append(str(class_names[lbl]))
    ax = plt.figure(num=plot_num, figsize=(8, 6), dpi=100).gca()
    for j in range(len(class_names)):
        plt.plot(smooth_curve(data_points[:, j]))

    plt.ylabel(axis_labels[0])
    plt.xlabel(axis_labels[1])
    plt.legend(legend_names)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    filename = os.path.join(eps_dir, axis_labels[0] + "_" + str(epoch) + "_Epochs.eps")
    plt.savefig(filename, format='eps', dpi=300, bbox_inches="tight")

    plt.show()


def plot_vol(vol_array, output_path):
    """
        save a file from slices of a volume array.
        If volume is int8, the function plots labelmap in color scale.
        otherwise the function consider the volume as a tomogram and plots in gray scale.

        inputs: vol_array: a 3D numpy array
                filename: '/path/to/output png file'
    """

    # Get central slices along each dimension:
    zindx = np.int(np.round(vol_array.shape[0]/2))
    yindx = np.int(np.round(vol_array.shape[1]/2))
    xindx = np.int(np.round(vol_array.shape[2]/2))

    xy_slice = vol_array[zindx, :, :]  # the xy plane
    zx_slice = vol_array[:, yindx, :]  # the zx plane
    zy_slice = vol_array[:, :, xindx]  # the zy plane

    if vol_array.dtype == np.int8:
        fig1 = plt.figure(num=1, figsize=(10, 10))
        plt.imshow(xy_slice, cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
        fig2 = plt.figure(num=2, figsize=(10, 5))
        plt.imshow(zx_slice, cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
        fig3 = plt.figure(num=3, figsize=(5, 10))
        plt.imshow(np.flipud(np.rot90(zy_slice)), cmap='jet', vmin=np.min(vol_array), vmax=np.max(vol_array))
    else:
        mu = np.mean(vol_array)  # mean of the volume/tomogram
        std = np.std(vol_array)  # standard deviation of the volume/tomogram
        fig1 = plt.figure(num=1, figsize=(10, 10))
        plt.imshow(xy_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)
        fig2 = plt.figure(num=2, figsize=(10, 5))
        plt.imshow(zy_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)
        fig3 = plt.figure(num=3, figsize=(5, 10))
        plt.imshow(zx_slice, cmap='gray', vmin=mu - 5 * std, vmax=mu + 5 * std)

    fig1.savefig(os.path.join(output_path, "segment/labelmap_xy_plane.png"))
    fig2.savefig(os.path.join(output_path, "segment/labelmap_zx_plane.png"))
    fig3.savefig(os.path.join(output_path, "segment/labelmap_zy_plane.png"))
    plt.show()



# export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64 #${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
