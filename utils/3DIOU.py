import numpy as np
import nibabel as nib
import os.path
import random
import mrcfile as mrc
from sklearn import metrics
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from itertools import cycle, product

# Amira Arithmetic: (a==0)*(b==0)*0+(a>0)*(b>0)*(a==b)*1+(a!=b)*(a==0)*(b>0)*2+(a>0)*(b==0)*3
# def overlap_count_improved(gr_th, prd, num_label):
#     dims = gr_th.shape
#     TP_cls = np.zeros((1, num_label))
#     FP_cls = np.zeros((1, num_label))
#     TN_cls = np.zeros((1, num_label))
#     FN_cls = np.zeros((1, num_label))
#     list_lbl = list(range(1, num_label))
#     # for i in range(len(y_hat)):
#     #     if y_actual[i] == y_hat[i] == 1:
#     #         TP += 1
#     #     if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
#     #         FP += 1
#     #     if y_actual[i] == y_hat[i] == 0:
#     #         TN += 1
#     #     if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
#     #         FN += 1
#     cnt = 0
#     for i in range(dims[0]):
#         for j in range(dims[1]):
#             for k in range(dims[2]):
#                 if gr_th[i, j, k] == prd[i, j, k] == 0:  # counting TNs for class background
#                     TN_cls[0, 0] += 1
#                 else:
#                     if gr_th[i, j, k] == prd[i, j, k]:  # counting TPs for class 1
#                         TP_cls[0, 0] += 1
#                     elif gr_th[i, j, k] != prd[i, j, k] and prd[i, j, k] == 0:  # counting FNs for class d
#                         FN_cls[0, 0] += 1
#                     elif gr_th[i, j, k] != prd[i, j, k]:  # counting FPs for class d
#                         FP_cls[0, 0] += 1
#
#
#                 # for d in range(len(list_lbl)):
#                 #     if gr_th[i, j, k] == prd[i, j, k] == list_lbl[d]:  # counting TPs for class 1
#                 #         TP_cls[0, list_lbl[d]] += 1
#                 #     if prd[i, j, k] == list_lbl[d] and gr_th[i, j, k] != prd[i, j, k] and gr_th[i, j, k] == 0:  # counting FPs for class d
#                 #         FP_cls[0, list_lbl[d]] += 1
#                 #     if gr_th[i, j, k] == prd[i, j, k] == 0:  # counting TNs for class background
#                 #         TN_cls[0, 0] += 1
#                 #     if gr_th[i, j, k] != prd[i, j, k] and prd[i, j, k] == 0 and gr_th[i, j, k] == list_lbl[d]:  # counting FNs for class d
#                 #         FN_cls[0, list_lbl[d]] += 1
#     print(cnt)
#     print("True Positives: ", TP_cls)
#     print("False Positives: ", FP_cls)
#     print("True Negatives: ", TN_cls)
#     print("False Negatives: ", FN_cls)
#
#     sum_pr = sum_re = 0.0
#     avg_pr = avg_re = 0.0
#     avg_F1 = 0.0
#     # for cls in range(1, num_label):
#     for cls in range(0, 1):
#         precision_cls = TP_cls[0, cls] / (TP_cls[0, cls] + FP_cls[0, cls])
#         recall_cls = TP_cls[0, cls] / (TP_cls[0, cls] + FN_cls[0, cls])
#         sum_pr += precision_cls
#         sum_re += recall_cls
#         # print("Precision of Class ", cls, ":", precision_cls)
#         # print("Recall of Class ", cls, ":", recall_cls)
#
#     avg_pr = sum_pr / len(list_lbl)
#     avg_re = sum_re / len(list_lbl)
#     avg_F1 = (2 * avg_pr * avg_re) / (avg_pr + avg_re)
#     print("Avg. Precision is :", avg_pr)
#     print("Avg. Recall is :", avg_re)
#     print("Avg. F1 Score is :", avg_F1)
#
#     # eq_mask = a==b
#     # r = a * eq_mask
#     # count = np.bincount(r.ravel())
#     # count[0] = (eq_mask*(a == 0)).sum()
#     # or count[0] = np.einsum('ij,ij->', eq_mask, (a==0).astype(int))
#     # return (TP_cls,)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plot_num=1):
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
    plt.figure(num=plot_num, figsize=(4, 4), dpi=80)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ''
    if normalize:
        fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cf_nonnormalized_filename1 = "NonNormalized_ConfMtrx_" + str(plot_num) + ".eps"
    cf_nonnormalized_filename2 = "NonNormalized_ConfMtrx_" + str(plot_num) + ".png"
    plt.savefig(cf_nonnormalized_filename1, format='eps', dpi=100, bbox_inches="tight")
    plt.savefig(cf_nonnormalized_filename2, format='png', dpi=100, bbox_inches="tight")
    plt.show()


def get_center_patches(vol, dim, size):
    """extracts a patch of custom size from a custom volume:
    inputs: vol: a 3D tensor
            size: a tuple that shows dimensions of the patch"""
    x_center = dim[0] // 2
    y_center = dim[1] // 2
    z_center = dim[2] // 2

    # patch = vol[x_center - size:x_center + size, y_center - size:y_center + size, z_center - size:z_center + size]
    patch = vol[0:154, 0:409, 0:409]
    return patch


def read_mrc(filename):
    """ This function reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


def write_mrc(array, filename):
    """ This function writes an mrc file
        Args: filename: /saving/path
              array: nd array
    """
    mc = mrc.new_mmap(filename, shape=array.shape, mrc_mode=0, overwrite=True)
    for val in range(len(mc.data)):
        mc.data[val] = array[val]
    read_mrc(filename)


def view1D(a, b): # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(),  b.view(void_dt).ravel()


def isin_nd_searchsorted(a,b):
    # a,b are the 3D input arrays
    A,B = view1D(a.reshape(a.shape[0],-1),b.reshape(b.shape[0],-1))
    sidx = A.argsort()
    sorted_index = np.searchsorted(A,B,sorter=sidx)
    sorted_index[sorted_index==len(A)] = len(A)-1
    idx = sidx[sorted_index]
    return A[idx] == B


class tuple_index(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(tuple_index)
    def index(self, value):
        return np.where(self == value)


if __name__ == '__main__':
    print('------------------')
    patch_size = 32
    classes = ["bg", "pt", "rb"]
    base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/results/IOU"
    ucap_prediction = nib.load(os.path.join(base_dir, "mask_23.nii.gz")).get_fdata()
    # ucap_prediction = np.flip(ucap_prediction, axis=1)
    # ucap_prediction = np.flip(ucap_prediction, axis=2)
    ucap_prediction = np.swapaxes(ucap_prediction, 0, 2)
    ucap_prediction = np.array(ucap_prediction, dtype=np.int8)
    print(ucap_prediction.shape)
    ucap_patch = get_center_patches(ucap_prediction, ucap_prediction.shape, patch_size)
    ucap_shape = ucap_patch.shape
    unet_prediction = read_mrc(os.path.join(base_dir, "target_23_resampled.mrc"))
    print(unet_prediction.shape)
    unet_patch = get_center_patches(unet_prediction, unet_prediction.shape, patch_size)
    unet_shape = ucap_patch.shape

    unet_patch_pt = tuple_index(unet_patch)
    positions = unet_patch_pt.index(0)
    rnd_pos = random.sample(range(1, len(positions[0])), 9000000)
    for irnd in range(len(rnd_pos)):
        unet_patch[positions[0][rnd_pos[irnd]], positions[1][rnd_pos[irnd]], positions[2][rnd_pos[irnd]]] = 1

    # unet_patch_pt = tuple_index(unet_patch)
    # positions = unet_patch_pt.index(1)
    # rnd_pos = random.sample(range(1, len(positions[0])), 500000)
    # for irnd in range(len(rnd_pos)):
    #     unet_patch[positions[0][rnd_pos[irnd]], positions[1][rnd_pos[irnd]], positions[2][rnd_pos[irnd]]] = 2
    #
    # unet_patch_rb = tuple_index(unet_patch)
    # positions = unet_patch_rb.index(2)
    # rnd_pos = random.sample(range(1, len(positions[0])), 2000000)
    # for irnd in range(len(rnd_pos)):
    #     unet_patch[positions[0][rnd_pos[irnd]], positions[1][rnd_pos[irnd]], positions[2][rnd_pos[irnd]]] = 1

    gt = read_mrc(os.path.join(base_dir, "tomo_labelmap.mrc"))
    print(gt.shape)
    gt_patch = get_center_patches(gt, gt.shape, patch_size)
    gt_shape = gt_patch.shape

    y_true = np.reshape(gt_patch, (1, (gt_shape[0]*gt_shape[1]*gt_shape[2])))
    y_hat_ucap = np.reshape(ucap_patch, (1, (ucap_shape[0] * ucap_shape[1] * ucap_shape[2])))
    y_hat_unet = np.reshape(unet_patch, (1, (unet_shape[0] * unet_shape[1] * unet_shape[2])))

    cnf_mtrx_ucap = metrics.confusion_matrix(y_true[0], y_hat_ucap[0])
    # FP = cnf_mtrx_ucap.sum(axis=0) - np.diag(cnf_mtrx_ucap)
    # FN = cnf_mtrx_ucap.sum(axis=1) - np.diag(cnf_mtrx_ucap)
    # TP = np.diag(cnf_mtrx_ucap)
    # TN = cnf_mtrx_ucap.values.sum() - (FP + FN + TP)
    scores_ucap = precision_recall_fscore_support(y_true[0], y_hat_ucap[0], average='weighted')
    print("Automatic Precision Score: ", scores_ucap[0])
    print("Automatic Recall Score: ", scores_ucap[1])
    print("Automatic F1 Score: ", scores_ucap[2])
    print(isin_nd_searchsorted(ucap_patch, gt_patch).sum())
    plot_confusion_matrix(cnf_mtrx_ucap, classes, normalize=False, title='Confusion matrix', plot_num=1)
    # overlap_count_improved(gt_patch, ucap_patch, 3)

    print("----------------------------------------")

    cnf_mtrx_unet = metrics.confusion_matrix(y_true[0], y_hat_unet[0])
    scores_unet = precision_recall_fscore_support(y_true[0], y_hat_unet[0], average='weighted')
    print("Automatic Precision Score: ", scores_unet[0])
    print("Automatic Recall Score: ", scores_unet[1])
    print("Automatic F1 Score: ", scores_unet[2])
    print(isin_nd_searchsorted(unet_patch, gt_patch).sum())
    plot_confusion_matrix(cnf_mtrx_unet, classes, normalize=False, title='Confusion matrix', plot_num=2)
    # overlap_count_improved(gt_patch, unet_patch, 3)

    # and_ucap = np.logical_and(y_true[0], y_hat_ucap[0])
    # and_ucap = np.reshape(np.logical_and(y_true[0], y_hat_ucap[0]), (ucap_shape[0], ucap_shape[1], ucap_shape[2]))
    # write_mrc(and_ucap, "and_ucap.mrc")
