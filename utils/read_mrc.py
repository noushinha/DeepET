import mrcfile as mrc
import sys
import numpy as np


def is_empty(arr, var):
    if arr.size == 0:
        print('array "' + var + '" is empty. Non empty array is expected.')
        sys.exit()


def is_file(filename):
    from pathlib import Path
    if not Path(filename).is_file():
        print('file "' + filename + '" does not exist. A valid file is required.')
        sys.exit()
    else:
        return 1


def read_mrc(filename):
    """ This function reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """
    if is_file(filename):
        with mrc.open(filename, mode='r+', permissive=True) as mc:
            # print(mc.print_header())
            # mc.update_header_from_data()
            # mc.update_header_stats()
            # # mc.header.exttyp = 'FEI1'
            # # mc.set_extended_header(mc.header)
            print(mc.print_header())
            mrc_tomo = mc.data

        # print(np.unique(mrc_tomo))
        print(mrc_tomo.shape)

        is_empty(mrc_tomo, 'mrc_tomo')
    return mrc_tomo



filename1 = '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_10/10.mrc'
# filename2 = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/0/masks/target_grandmodel_0.mrc'
read_mrc(filename1)
# read_mrc(filename2)

print(mrc.validate(filename1))
# print(mrc.validate(filename1))
