# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MaxNLocator
# import pandas as pd
# import os
#
# epoch = 100
# legend_names = []
# class_names = ['bg', '1bxn', '1qvr', '1s3x', '1u6g', '2cg9', '3cf3', '3d2f', '3gl1', '3h84', '3qm1', '4b4t', '4d8q']
# # axis_labels = ['Epochs', 'F1 Score']
# # axis_labels = ['Epochs', 'Recall']
# axis_labels = ['Epochs', 'Precision']
# # f1_scores = pd.read_csv('/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Validation_Recall_Details.csv', header=None)
# # f1_scores = pd.read_csv('/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Validation_F1Score_Details.csv', header=None)
# f1_scores = pd.read_csv('/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Validation_Precision_Details.csv', header=None)
# f1_scores = f1_scores.to_numpy()
# dims = f1_scores.shape
# sum2 = np.zeros((100, 13))
#
# def smooth_curve(points, factor=0.8):
#     """ This function smooths the fluctuation of a plot by
#         calculating the moving average of the points based on factor
#     """
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
# for i in range(dims[0]):
#     sum1 = np.zeros((13))
#     for j in range(dims[1]):
#         scores = f1_scores[i][j].replace("\n ", "         ")
#         scores = scores.replace("         ", ",")
#         scores = scores.replace("        ]", "")
#         scores = scores.replace("       ]", "")
#         scores = scores.replace("        ", ",")
#         scores = scores.replace("[", "")
#         scores = scores.replace("]", "")
#         scores = scores.replace("       ", ",")
#         scores = scores.replace("      ", ",")
#         scores = scores.replace("     ", ",")
#         scores = scores.replace("    ", ",")
#         scores = scores.replace("   ", ",")
#         scores = scores.replace("  ", ",")
#         scores = scores.replace(" ", ",")
#         scores = scores.split(",")
#
#         scores = [float(x) for x in scores]
#         sum1 = sum1 + scores
#     sum2[i, :] = sum1
#
#
# data_points = sum2
# epoch = data_points.shape[0]
#
# for lbl in range(len(class_names)):
#     legend_names.append(str(class_names[lbl]))
# ax = plt.figure(num=1, figsize=(8, 6), dpi=100).gca()
# for j in range(len(class_names)):
#     plt.plot(smooth_curve(data_points[:, j]))
#
# plt.ylabel(axis_labels[0])
# plt.xlabel(axis_labels[1])
# plt.legend(legend_names)
# plt.grid()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
# # filename1 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Recall_100_Epochs.eps"
# # filename2 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Recall_100_Epochs.png"
#
# # filename1 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/F1 Score_100_Epochs.eps"
# # filename2 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/F1 Score_100_Epochs.png"
#
# filename1 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Precision_100_Epochs.eps"
# filename2 = "/mnt/Data/Cryo-ET/DeepET/data2/results/8VOVDRAT9LTM5W8S/Precision_100_Epochs.png"
#
# plt.savefig(filename1, format='eps', dpi=500, bbox_inches="tight")
# plt.savefig(filename2, format='png', dpi=500, bbox_inches="tight")
# plt.show()
import sys
import mrcfile


def is_file(filename):
    from pathlib import Path
    if not Path(filename).is_file():
        print('file "' + filename + '" does not exist. A valid file is required.')
        sys.exit()
    else:
        return 1


def is_empty(arr, var):
    if arr.size == 0:
        print('array "' + var + '" is empty. Non empty array is expected.')
        sys.exit()


def read_mrc(filename):
    """ This function reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """
    if is_file(filename):
        import mrcfile as mrc
        with mrc.open(filename, mode='r+', permissive=True) as mc:
            mc.update_header_from_data()
            mrc_tomo = mc.data
        is_empty(mrc_tomo, 'mrc_tomo')
    return mrc_tomo


def write_mrc(array, filename):
    """ This function writes an mrc file
        Args: filename: /saving/path
              array: nd array
    """
    import mrcfile as mrc
    with mrc.new(filename, overwrite=True) as mc:
        mc.set_data(array)


# path_data = ['/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_0.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_1.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_2.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_3.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_4.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_5.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_6.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_7.mrc',
#              '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_8.mrc']
#
#
# data_list = []
# target_list = []
# for idx in range(0, len(path_data)):
#     data = read_mrc(path_data[idx])
#     data = data[156:356, :, :]
#     write_mrc(data, '/mnt/Data/Cryo-ET/DeepET/data2/images/reconstruction_model_0' + str(idx) + '.mrc')
#

data = read_mrc('/mnt/Data/Cryo-ET/DeepET/data2/results/evaluation/reconstruction_model_9.mrc')
data = data[156:356, :, :]
write_mrc(data, '/mnt/Data/Cryo-ET/DeepET/data2/results/evaluation/reconstruction_model_09.mrc')