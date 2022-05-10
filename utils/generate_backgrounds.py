import os
import mrcfile as mrc
import numpy as np


def read_mrc(filename):
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    return mrc_tomo


def write_mrc(array, filename):
    with mrc.new(filename, overwrite=True) as mc:
        mc.set_data(array)
    read_mrc(filename)


tomoid = 5
tomo_id = "0" + str(tomoid)

base_dir = "/mnt/Data/Cryo-ET/DeepET/data2/DEEPETGTMasks/"
output = "/mnt/Data/Cryo-ET/DeepET/data2/DeepET_Tomo_Masks_1Class/"
tomo_name = base_dir + 'reconstruction_model_' + str(tomo_id) + '.mrc'
mask_name = base_dir + 'target_grandmodel_' + str(tomoid) + '.mrc'
tomo = read_mrc(tomo_name)
mask = read_mrc(mask_name)

print(tomo.shape)
print(mask.shape)

label_indices = np.argwhere(mask != 0)
bg_indx = np.argwhere(mask == 0)
cnt = 0

for indx in label_indices:
    # if mask[tuple(indx)] != 6 and mask[tuple(indx)] != 11 and mask[tuple(indx)] != 12:
    if mask[tuple(indx)] != 6:
        tomo[tuple(indx)] = tomo[tuple(bg_indx[cnt])]
        # tomo[tuple(indx)] = tomo[tuple(bg_indx[cnt])]
        cnt = cnt + 1

write_mrc(tomo, os.path.join(output, "reconstruction_model_" + str(tomo_id) + ".mrc"))
