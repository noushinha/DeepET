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
from pathlib import Path
from sys import platform
import sys

SOFTNAME = "DEEP ET"
VERSION = 1.0
ROOT_DIR = Path(__file__).parent.parent
print(ROOT_DIR)
print(SOFTNAME, " ", VERSION)

OS = "Linux"
if platform == "linux" or platform == "linux2":
    OS = "Linux"
    OS_path_separator = "/"
elif platform == "win32":
    OS = "Windows"
    OS_path_separator = "\\"
else:
    print('This software is not supported for ' + platform + ' systems.')
    sys.exit()


# ROOT_DIR =  '/mnt/Data/Cryo-ET/DeepTomo/'
# ROOT_DIR = "C:\\Users\\Asus\\Desktop\\DeepTomo"