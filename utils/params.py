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
from collections import OrderedDict
from pycm import ConfusionMatrix

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
    OS_path_separator = "/"  # "\\"?
else:
    print('This software is not supported for ' + platform + ' systems.')
    sys.exit()

# SHREC with 12 classes
class_names_list = ["1bxn", "1qvr", "1s3x", "1u6g", "2cg9", "3cf3", "3d2f", "3gl1", "3h84", "3qm1", "4b4t", "4d8q"]
class_names = {0: "bg", 1: "1bxn", 2: "1qvr", 3: "1s3x", 4: "1u6g", 5: "2cg9", 6: "3cf3",
               7: "3d2f", 8: "3gl1", 9: "3h84", 10: "3qm1", 11: "4b4t", 12: "4d8q"}
reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
                                    "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})
class_radius = [0, 6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8]

# SHREC with 2 classes
# class_radius = [0, 10, 8]
# class_names_list = ["4b4t", "4d8q"]
# class_names = {0: "bg", 1: "4b4t", 2: "4d8q"}
# reversed_class_names = OrderedDict({"bg": 0, "4b4t": 1, "4d8q": 2})

# Invitro with 2 classes
# class_radius = [0, 12, 8]
# class_names_list = ["pt", "rb"]
# class_names = {0: "bg", 1: "pt", 2: "rb"}
# reversed_class_names = OrderedDict({"bg": 0, "pt": 1, "rb": 2})
