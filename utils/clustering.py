import os
import numpy as np
import mrcfile as mrc
import nibabel as nib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_center_patches(vol, dim, size):
    """extracts a patch of custom size from a custom volume:
    inputs: vol: a 3D tensor
            size: a tuple that shows dimensions of the patch"""
    x_center = dim[0] // 2
    y_center = dim[1] // 2
    z_center = dim[2] // 2

    patch = vol[x_center - size:x_center + size, y_center - size:y_center + size, z_center - size:z_center + size]
    # patch = vol[0:154, 0:409, 0:409]
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


def bin_tomo(scoremap_tomo):
    from skimage.measure import block_reduce

    bd0 = int(np.ceil(scoremap_tomo.shape[0] / 2))
    bd1 = int(np.ceil(scoremap_tomo.shape[1] / 2))
    bd2 = int(np.ceil(scoremap_tomo.shape[2] / 2))
    new_dim = (bd0, bd1, bd2, 3)
    binned_scoremap = np.zeros(new_dim)

    for cnum in range(3):
        # averaging over 2x2x2 patches to subsample
        # we divided the dimension by half so each size dimension is subsampled by 2,
        # also apply for each class dimension
        binned_scoremap[:, :, :, cnum] = block_reduce(scoremap_tomo[:, :, :, cnum], (2, 2, 2), np.mean)

    return binned_scoremap


if __name__ == '__main__':
    print('------------------')
    patch_size = 8
    classes = ["bg", "pt", "rb"]
    base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/results/IOU"

    ucap_prediction = nib.load(os.path.join(base_dir, "mask_23.nii.gz")).get_fdata()
    ucap_prediction = np.swapaxes(ucap_prediction, 0, 2)
    ucap_prediction = np.array(ucap_prediction, dtype=np.int8)
    print(ucap_prediction.shape)
    ucap_patch = get_center_patches(ucap_prediction, ucap_prediction.shape, patch_size)
    ucap_shape = ucap_patch.shape

    unet_prediction = read_mrc(os.path.join(base_dir, "target_23_resampled.mrc"))
    print(unet_prediction.shape)
    unet_patch = get_center_patches(unet_prediction, unet_prediction.shape, patch_size)
    unet_shape = ucap_patch.shape

    gt = read_mrc(os.path.join(base_dir, "tomo_labelmap.mrc"))
    print(gt.shape)
    gt_patch = get_center_patches(gt, gt.shape, patch_size)
    gt_shape = gt_patch.shape

    y_true = np.reshape(gt_patch, (gt_shape[0], (gt_shape[1] * gt_shape[2])))
    y_hat_ucap = np.reshape(ucap_patch, (ucap_shape[0], (ucap_shape[1] * ucap_shape[2])))
    y_hat_unet = np.reshape(unet_patch, (unet_shape[0], (unet_shape[1] * unet_shape[2])))

    from mpl_toolkits.mplot3d import Axes3D
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(y_hat_ucap)
    y_kmeans = kmeans.predict(y_hat_ucap)
    plt.scatter(y_hat_ucap[:, 0], y_hat_ucap[:, 1], c=y_kmeans, s=50, cmap='viridis')
    #
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
