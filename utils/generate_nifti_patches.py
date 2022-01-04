#!/usr/bin/env python3


import os
import re
import sys
import json
from glob import glob
import numpy as np
import nibabel as nib
import mrcfile as mrc
from lxml import etree

base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/DEEPETGTMasks/"
output_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3"
classNum = 3
patch_size = 64
patches_tomos = []
patches_masks = []


def read_mrc(filename):
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


def read_xml2(filename):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    for p in range(len(objl_xml)):
        object_id = objl_xml[p].get('obj_id')
        tomo_idx = objl_xml[p].get('tomo_idx')
        lbl = objl_xml[p].get('class_label')
        x = objl_xml[p].get('x')
        y = objl_xml[p].get('y')
        z = objl_xml[p].get('z')

        if object_id is not None:
            object_id = int(object_id)
        else:
            object_id = p
        if tomo_idx is not None:
            tomo_idx = int(tomo_idx)
        add_obj(obj_list, tomo_idx=tomo_idx, obj_id=object_id, label=int(lbl), coord=(float(z), float(y), float(x)))
    return obj_list


def add_obj(obj_list, label, coord, obj_id=None, tomo_idx=None, c_size=None):
    obj = {
        'tomo_idx': tomo_idx,
        'obj_id': obj_id,
        'label': label,
        'x': coord[2],
        'y': coord[1],
        'z': coord[0],
        'c_size': c_size
    }

    obj_list.append(obj)
    return obj_list


def get_patch_position(tomodim, p_in, obj, shiftr):
    x = int(obj['x'])
    y = int(obj['y'])
    z = int(obj['z'])

    # Add random shift to coordinates:
    x = x + np.random.choice(range(-shiftr, shiftr + 1))
    y = y + np.random.choice(range(-shiftr, shiftr + 1))
    z = z + np.random.choice(range(-shiftr, shiftr + 1))

    # Shift position if passes the borders:
    if x < p_in:
        x = p_in
    if y < p_in:
        y = p_in
    if z < p_in:
        z = p_in

    if x > tomodim[2] - p_in:
        x = tomodim[2] - p_in
    if y > tomodim[1] - p_in:
        y = tomodim[1] - p_in
    if z > tomodim[0] - p_in:
        z = tomodim[0] - p_in

    return x, y, z


def int2str(n):
    strn = ""
    if 0 <= n < 10:
        strn = "000" + str(n)
    elif 10 <= n < 100:
        strn = "00" + str(n)
    elif 100 <= n < 1000:
        strn = "0" + str(n)
    else:
        strn = str(n)
    return str(strn)

list_tomoID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(9, 10):
    tomo_id = "0" + str(i)
    img_mrc = base_dir + '/reconstruction_model_' + str(tomo_id) + '.mrc'
    msk_mrc = base_dir + 'target_grandmodel_' + str(i) + '.mrc'
    # img_mrc = base_dir + '/' + str(list_tomoID[i]) + '_resampled.mrc'
    # msk_mrc = base_dir + 'target_' + str(list_tomoID[i]) + '_resampled.mrc'
    tomo = read_mrc(img_mrc)
    mask = read_mrc(msk_mrc)
    mask = mask[:, 0:512, 0:512]

    # check if the tomogram and its mask are of the same size
    if tomo.shape != mask.shape:
        print("the tomogram and the target must be of the same size. " +
              str(tomo.shape) + " is not equal to " + str(mask.shape) + ".")
        sys.exit()

    patches_tomos.append(tomo)
    patches_masks.append(mask)

list_annotations = read_xml2(os.path.join(base_dir, "object_list_test.xml"))
mid_dim = int(np.floor(patch_size / 2))
print(len(list_annotations))
cnt = 0
tomo_idx_prev = -1
for i in range(0, len(list_annotations)):  #
    # find the tomo
    tomo_idx = int(list_annotations[i]['tomo_idx'])

    if tomo_idx_prev != tomo_idx:
        cnt = 0
    # read_tomo
    sample_tomo = patches_tomos[0]
    sample_mask = patches_masks[0]

    # correct positions
    x, y, z = get_patch_position(patches_tomos[0].shape, mid_dim, list_annotations[i], 13)

    # extract the patch from tomo and mask
    patch_tomo = sample_tomo[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]
    # patch_tomo = (patch_tomo - np.mean(patch_tomo)) / np.std(patch_tomo)
    patch_mask = sample_mask[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]

    # save the extracted patches and their masks as nifti
    empty_header_tomo = nib.Nifti1Header()
    empty_header_tomo.get_data_shape()
    nifti_tomo = nib.Nifti1Image(patch_tomo, np.eye(4))

    # saving the tomo
    patch_tomo_name = "imagesTs/patch_t" + str(tomo_idx) + "_p" + int2str(cnt) + ".nii.gz"
    # if tomo_idx == 3:
    #     patch_tomo_name = "imagesTs/patch_t" + str(tomo_idx) + "_p" + int2str(cnt) + ".nii.gz"
    patch_tomo_name = os.path.join(output_dir, patch_tomo_name)
    nib.save(nifti_tomo, patch_tomo_name)
    print(patch_tomo_name + " is saved")

    empty_header_mask = nib.Nifti1Header()
    empty_header_mask.get_data_shape()
    nifti_mask = nib.Nifti1Image(np.array(patch_mask, dtype=np.uint8), np.eye(4))

    # saving the mask
    # if tomo_idx != 3:
    patch_mask_name = "labelsTs/patch_m" + str(tomo_idx) + "_c" + int2str(cnt) + ".nii.gz"
    patch_mask_name = os.path.join(output_dir, patch_mask_name)
    nib.save(nifti_mask, patch_mask_name)
    print(patch_mask_name + " is saved")
    tomo_idx_prev = tomo_idx
    cnt = cnt + 1
    # patch_mask_onehot = to_categorical(patch_mask, classNum)

# generating a json.dataset file similar to that of hippocampus dataset
# base_dir = "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3"
# #
# data = dict()
# data = {"name": "SHREC",
#         "description": "molecular Structure Segmentation",
#         "reference": " Utrecht University",
#         "licence": "CC-BY-SA 4.0",
#         "release": "shrec subdata",
#         "tensorImageSize": "3D",
#         "modality": {"0": "tomogram"},
#         "labels": {"0": "bg", "1": "1bxn", "2": "1qvr", "3": "1s3x", "4": "1u6g", "5": "2cg9", "6": "3cf3",
#                    "7": "3d2f", "8": "3gl1", "9": "3h84", "10": "3qm1", "11": "4b4t", "12": "4d8q"},
#         "numTraining": 2498,
#         "numTest": 2449,
#         }
#
# list_tr_tomos_IDs = glob(os.path.join(base_dir, "imagesTr/*.nii.gz"))
# list_tr_masks_IDs = glob(os.path.join(base_dir, "labelsTr/*.nii.gz"))
# list_ts_tomos_IDs = glob(os.path.join(base_dir, "imagesTs/*.nii.gz"))
#
# list_tr_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
# list_tr_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))
# list_ts_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
#
# training_item = []
# for t in range(len(list_tr_tomos_IDs)):
#         tomo = list_tr_tomos_IDs[t]
#         tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3", ".")
#         mask = list_tr_masks_IDs[t]
#         mask = str.replace(mask, "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3", ".")
#
#         strdict = {"image": tomo, "label": mask}
#         training_item.append(strdict)
# data["training"] = training_item
#
# test_item = []
# for t in range(len(list_ts_tomos_IDs)):
#         tomo = list_ts_tomos_IDs[t]
#         tomo = str.replace(tomo, "/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3", ".")
#         test_item.append(tomo)
# data["test"] = test_item
#
# print(data)
# with open('/mnt/Data/Cryo-ET/3D-UCaps/data/shrec3/dataset.json', 'w') as outfile:
#      json.dump(data, outfile, indent=2)
#      outfile.write('\n')
#
