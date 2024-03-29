# ============================================================================================
# DeepET - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2023 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================
from pathlib import Path
from sys import platform
import sys
from collections import OrderedDict
from pycm import ConfusionMatrix
# Fhsr5BuLa9JxRYE
SOFTNAME = "DEEP ET"
VERSION = 2.0
ROOT_DIR = Path(__file__).parent.parent
# print(ROOT_DIR)
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
# class_names_list = ["1bxn", "1qvr", "1s3x", "1u6g", "2cg9", "3cf3", "3d2f", "3gl1", "3h84", "3qm1", "4b4t", "4d8q"]
# class_names = {0: "bg", 1: "1bxn", 2: "1qvr", 3: "1s3x", 4: "1u6g", 5: "2cg9", 6: "3cf3",
#                7: "3d2f", 8: "3gl1", 9: "3h84", 10: "3qm1", 11: "4b4t", 12: "4d8q"}
# reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
#                                     "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})
# class_radius = [0, 6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8]

# ********************** SHREC19 Dataset **********************
# SHREC19 with class 4D8Q
# class_radius = [0, 8]
# class_names_list = ["4d8q"]
# class_names = {0: "bg", 1: "4d8q"}
# reversed_class_names = OrderedDict({"bg": 0, "4d8q": 1})

# SHREC19 with class 1BXN
# class_radius = [0, 6]
# class_names_list = ["1bxn"]
# class_names = {0: "bg", 1: "1bxn"}
# reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1})

# SHREC19 with class 3GL1
# class_radius = [0, 4]
# class_names_list = ["3gl1"]
# class_names = {0: "bg", 1: "3gl1"}
# reversed_class_names = OrderedDict({"bg": 0, "3gl1": 1})

# SHREC19 with class 3GL1
# class_radius = [0, 8, 6, 4]
# class_names_list = ["4d8q", "1bxn", "3gl1"]
# class_names = {0: "bg", 1: "4d8q", 2: "1bxn", 3: "3gl1"}
# reversed_class_names = OrderedDict({"bg": 0, "4d8q": 1, "1bxn": 2, "3gl1": 3})

# ********************** Invitro Dataset **********************
# Invitro with 2 classes
class_radius = [0, 10, 13]
class_names_list = ["pt", "rb"]
class_names = {0: "bg", 1: "pt", 2: "rb"}
reversed_class_names = OrderedDict({"bg": 0, "pt": 1, "rb": 2})

# Invitro with class proteasome
# class_radius = [0, 10]
# class_names_list = ["pt"]
# class_names = {0: "bg", 1: "pt"}
# reversed_class_names = OrderedDict({"bg": 0, "pt": 1})


# Invitro with class ribosome
# class_radius = [0, 13]
# class_names_list = ["rb"]
# class_names = {0: "bg", 1: "rb"}
# reversed_class_names = OrderedDict({"bg": 0, "rb": 1})

# class_radius = [0, 10, 13, 10, 13]
# class_names_list = ["pt", "rb", "ptsu", "rbsu"]
# class_names = {0: "bg", 1: "pt", 2: "rb", 3: "ptsu", 4: "rbsu"}
# reversed_class_names = OrderedDict({"bg": 0, "pt": 1, "rb": 2, "ptsu": 1, "rbsu": 2})