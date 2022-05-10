# imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#Variables
Dataset = "SAVEE"
base_dir = "/media/Data/IEEE Transaction on Affective Computing/Result/"
base = "VideoBase"
spec = "Audio"
face = "Video"
csv = "feature"
num_samples=6
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


# variables
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
spec_dir = os.path.join(base_dir, Dataset, base, spec, "0.44", csv)
face_dir = os.path.join(base_dir, Dataset, base, face, "0.92", csv)

test_spec_filename = "Test_Spec_fc6_Layer_Features.npy"
test_face_filename = "Test_Face_fc6_Layer_Features.npy"
test_spec_data = np.load(os.path.join(spec_dir, test_spec_filename))
test_face_data = np.load(os.path.join(face_dir, test_face_filename))

spec_data = []
spec_zeros = np.count_nonzero(test_spec_data, axis=0)
for i in range(0, 64):
    if(spec_zeros[i] != 0):
        spec_data.append(test_spec_data[:, i])

spec_data = np.array(spec_data)
spec_data = spec_data.transpose()
# print(spec_data.shape)

face_data = []
face_zeros = np.count_nonzero(test_face_data, axis=0)
for i in range(0,64):
    if(face_zeros[i] != 0):
        face_data.append(test_face_data[:,i])
face_data = np.array(face_data)
face_data = face_data.transpose()
# print(face_data.shape)


# pd.DataFrame(spec_data).to_csv(os.path.join(spec_dir, "Test_Spec_fc6_Layer_Features.csv"), index=False)
# pd.DataFrame(face_data).to_csv(os.path.join(face_dir, "Test_Face_fc6_Layer_Features.csv"), index=False)

# Univariate Histograms
def Univariate_Histograms(filename, title, size):
    names = []
    for i in range(1,size):
        names.append(str(i))
    # data = pd.DataFrame(X)
    names = names.sort()
    data = pd.read_csv(filename, names=names)
    # data.columns = pd.CategoricalIndex(names, ordered=True)
    data.hist(bins=15, color="g")
    # ax = plt.gca()
    # ax.grid(which='major', axis='both', linestyle='--', color="#e1e1e1")
    # plt.title(title)
    plt.show()

# Univariate_Histograms(os.path.join(face_dir, "Test_Face_fc6_Layer_Features.csv"), "Video histograms", face_data.shape[1])
# Univariate_Histograms(os.path.join(spec_dir, "Test_Spec_fc6_Layer_Features.csv"), "Audio histograms", spec_data.shape[1])


# univariate density plots

def Univariate_Density(filename, title, size):
    names = []
    for i in range(1, size):
        names.append(str(i))
    # data = pd.DataFrame(X)
    names = names.sort()
    data = pd.read_csv(filename, names=names)
    if(divmod(size, 3)[1] == 0):
        rows = 3
    else:
        rows = int(math.ceil(size / 3)) + 1

    data.plot(kind='density', subplots=True, layout=(rows, 3), sharex=False, legend=False)

    plt.show()
#
# Univariate_Density(os.path.join(face_dir, "Test_Face_fc6_Layer_Features.csv"), "Video histograms", face_data.shape[1])
# Univariate_Density(os.path.join(spec_dir, "Test_Spec_fc6_Layer_Features.csv"), "Audio histograms", spec_data.shape[1])

# univariate Box and Whisker plots
def Box_Whisker(filename, title, size):
    names = []
    for i in range(1, size):
        names.append(str(i))
    # data = pd.DataFrame(X)
    names = names.sort()
    data = pd.read_csv(filename, names=names)
    if(divmod(size, 3)[1] == 0):
        rows = 3
    else:
        rows = int(math.ceil(size / 3)) + 1

    data.plot(kind='box', subplots=True, layout=(rows, 3), sharex=False, sharey=False)
    plt.show()

# Box_Whisker(os.path.join(face_dir, "Test_Face_fc6_Layer_Features.csv"), "Video histograms", face_data.shape[1])
# Box_Whisker(os.path.join(spec_dir, "Test_Spec_fc6_Layer_Features.csv"), "Audio histograms", spec_data.shape[1])

# corrlation matrix
def Correlation_Matrix(filename, title, size):
    names = []
    for i in range(1, size):
        names.append(str(i))
    # data = pd.DataFrame(X)
    names = names.sort()
    data = pd.read_csv(filename, names=names)
    correlations = data.corr()
    print(correlations)
    df = pd.DataFrame(data)
    df.to_csv("SAVEE_Corrlation_Matrix_VideoBase_Features.csv")

    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,23,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # ax.set_xticklabels(names)
    # ax.set_yticklabels(names)
    plt.show()

test_data = np.hstack((face_data, spec_data))
pd.DataFrame(test_data).to_csv(os.path.join(face_dir, "Test_data.csv"), index=False)
Correlation_Matrix(os.path.join(face_dir, "Test_data.csv"), "Audio histograms", test_data.shape[1])

# Scatter Matrix
def Scatter_Matrix(filename, title, size):
    names = []
    for i in range(1, size):
        names.append(str(i))
    # data = pd.DataFrame(X)
    names = names.sort()
    data = pd.read_csv(filename, names=names)
    scatter_matrix(data)
    plt.show()

# Scatter_Matrix(os.path.join(face_dir, "Test_data.csv"), "Audio histograms", test_data.shape[1])
