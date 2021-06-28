# ============================================================================================
# DeepTomo - a deep learning framework for segmentation and classification of
#                  macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi, Willy (Daniel team)
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================
import pandas as pd
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from pathlib import Path
# from deeptomo.utils import params


def display(message):
    print(message)


def is_list(v, varname):
    if type(v) != list:
        display('DeepFinder message: variable "' + varname + '" is ' + str(type(v)) + '. Expected is list.')
        sys.exit()

def is_3D_nparray(v, varname):
    if type(v)!=np.ndarray:
        display('DeepFinder message: variable "'+varname+'" is '+str(type(v))+'. Expected is numpy array.')
        sys.exit()
    if len(v.shape)!=3:
        display('DeepFinder message: variable "'+varname+'" is a '+str(len(v.shape))+'D array. Expected is a 3D array.')
        sys.exit()

def file_attributes(Qtfile):
    """ This function receives a QtFile object that is selected through browse button and
        returns file path and file type of it.
        Args: Qtfile (an object returned from QFileDialog)
        Returns: string1 and string2: file path and file type
    """
    try:
        filetype = os.path.splitext(Qtfile[0])[1][1:]
        filename = str(Qtfile[0])
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return filename, filetype


def throwErr(ErrType='None'):
    errval = ErrType.split(":")
    if errval[0] == 'ext':
        print('DeepTomo message: The file extension {extension} is not supported.'.format(extension=errval[1]))

def read_xml(filename):
    """ This function receives the xml file path, reads it and returns an object
        returns file path and file type of it.
        Args: string (an string containing file path/file name)
        Returns: 2D array: a panda dataframe of labels
    """
    from lxml import objectify
    data = []
    cols = []

    xml_data = objectify.parse(filename)  # Parse XML data
    root = xml_data.getroot()  # Root element
    firstchild = root.getchildren()[0]
    all_attributes = list(firstchild.iter())
    cols = [element.tag for element in all_attributes]
    cols = cols[1:] # drop the first tag name because it is the name of the child itself and not the attributes

    for i in range(len(root.getchildren())):
        child = root.getchildren()[i]
        data.append([subchild.text for subchild in child.getchildren()])

    df = pd.DataFrame(data)  # Create DataFrame and transpose it
    df.columns = cols  # Update column names

    return df


def read_starfile(filename):
    """ This function receives a QtFile object that is selected through browse button and
        returns file path and file type of it.
        Args: Qtfile (an object returned from QFileDialog)
        Returns: string1 and string2: file path and file type
    """
    import starfile
    content = starfile.open(filename)
    # rlnImageName', 'rlnCoordinateX', 'rlnCoordinateY',
    return content

def read_mrc(filename):
    """ This function reads an mrc file and returns the 3D array
        Args: filename: path to mrc file
        Returns: 3d array
    """
    import mrcfile as mrc
    with mrc.open(filename, permissive=True) as mc:
        array = mc.data
    return array


def write_mrc(array, filename):
    """ This function writes an mrc file
        Args: filename: /saving/path
              array: nd array
    """
    import mrcfile as mrc
    with mrc.new(filename, overwrite=True) as mc:
        mc.set_data(array)

def get_coords(dim):
    """ This function receives dimensions of a 3D volume and returns center slices.
        Args: dim: the 3D volume shape(tomogram - e.g an mrc file)
        Returns: z, y, x center coordination
    """
    x = np.int(np.round(dim[1] / 2))
    y = np.int(np.round(dim[2] / 2))
    z = np.int(np.round(dim[0] / 2))
    return x, y, z

def get_planes(vol):
    """ This function receives a 3D volume and returns orthoclices of xy, zx, and zy planes.
        Args:
            vol : the 3D volume (tomogram - e.g an mrc file)
            coords : look get_coords
                     coords[0] -> z, coords[1] -> y, coords[2] -> x
        Returns: xy_plane, zx_plane, zy_plane: orthoslices of planes
    """
    x, y, z = get_coords(vol.shape)
    xy_plane = vol[z, :, :]
    zx_plane = vol[:, y, :]
    zy_plane = vol[:, :, x]
    return xy_plane, zx_plane, zy_plane


def normalize_image(img):
    """ This function Normalizes the norm or value range of an array.
        this is required for having colorful circles on the grayscale images
        Args: img: image nd.array that has a range of value like [-1, 1] or [0,1]
        Returns: image nd.array with range [0, 255]
    """
    # np.round((xy_plane + 1) * 255 / 2)
    img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img = img.astype('float32')
    return img

def g2c_image(img):
    """ This function converts the image from grayscale to RGB
        this is required for having colorful circles on the grayscale images
        Args: img: image nd.array that has a range of value [0, 255]
        Returns: image nd.array with range [0, 255]
    """
    if len(img.shape) <= 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def color_pallete(class_num, class_name):
    # boxcolor = dict.fromkeys(class_name, 0.0)
    # for i in class_name:
    #     boxcolor[i] = np.random.randint(low=0, high=255)
    boxcolor = np.random.permutation(255)[:class_num]
    # for i in range(class_num):
    #     boxcolor.append(np.random.randint(low=0, high=255))


    return boxcolor


def radius_list(class_name, radiuslisttext):
    radiuslist = radiuslisttext.split(",")
    class_radlist = dict.fromkeys(class_name, 0)
    cnt = 0
    for i in class_name:
        class_radlist[i] = int(radiuslist[cnt])
        cnt = cnt + 1

    return class_radlist


def create_sphere(dim, r):
    C = np.floor((dim[0]/2, dim[1]/2, dim[2]/2))
    x,y,z = np.meshgrid(range(dim[0]), range(dim[1]), range(dim[2]))

    sphere = ((x - C[0])/r)**2 + ((y - C[1])/r)**2 + ((z - C[2])/r)**2
    sphere = np.int8(sphere <= 1)
    return sphere


def generate_with_spheres(objl, target_array, class_radilist):
    """Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'.

    Args:
        objl: array of coordination
        target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
            index order of array should be [z,y,x]
        radius_list (list of int): contains sphere radii per class (in voxels).
            The radii order in list should correspond to the class label.
            For ex: 1st element of list -> sphere radius for class 1, 2nd element of list -> sphere radius for class 2 etc.

    Returns:
        3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
    """
    radius_vals = list(class_radilist.values())
    Rmax = max(radius_vals)
    # dim = [2*Rmax, 2*Rmax, 2*Rmax]
    ref_list = []
    for idx in range(len(radius_vals)):
        dim = [2 * radius_vals[idx], 2 * radius_vals[idx], 2 * radius_vals[idx]]
        my_sphere = create_sphere(dim, radius_vals[idx])
        ref_list.append(my_sphere)
    target_array = generate_with_shapes(objl, target_array, ref_list, class_radilist)
    return target_array


def generate_with_shapes(objl, target_array, ref_list, class_radilist):
    """Generates segmentation targets from object list. Here macromolecules are annotated with their shape.

    Args:
        objl (list of dictionaries): Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
        target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
            index order of array should be [z,y,x]
        ref_list (list of 3D numpy arrays): These reference arrays are expected to be cubic and to contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
            The references order in list should correspond to the class label.
            For ex: 1st element of list -> reference of class 1; 2nd element of list -> reference of class 2 etc.

    Returns:
        3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
    """
    # is_list(objl, 'objl')
    is_3D_nparray(target_array, 'target_array')
    is_list(ref_list, 'ref_list')
    boxcolor = color_pallete(len(class_radilist.keys()), list(class_radilist.keys()))

    N = objl.shape[0]
    dim = target_array.shape
    for row in range(N):
        lbl = int(tuple(class_radilist.keys()).index(objl[row][-1]))
        x = int(objl[row][3])
        y = int(objl[row][2])
        z = int(objl[row][1])
        display('Annotating point ' + str(row + 1) + ' / ' + str(N) +
                ' with class ' + str(objl[row][-1]) +
                ' and color ' + str(boxcolor[lbl]))

        ref = ref_list[lbl - 1]
        centeroffset = np.int(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

        # Get the coordinates of object voxels in target_array
        obj_voxels = np.nonzero(ref == 1)
        x_vox = obj_voxels[2] + x - centeroffset #+1
        y_vox = obj_voxels[1] + y - centeroffset #+1
        z_vox = obj_voxels[0] + z - centeroffset #+1

        for idx in range(x_vox.size):
            xx = x_vox[idx]
            yy = y_vox[idx]
            zz = z_vox[idx]
            if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds

                target_array[zz, yy, xx] = boxcolor[lbl]

    return np.int8(target_array)


# Writes an image file containing ortho-slices of the input volume. Generates same visualization as matlab function
# 'tom_volxyz' from TOM toolbox.
# If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
# Else, it assumes that the volume is tomographic data, and plots in gray scale.
# INPUTS:
#   vol     : 3D numpy array
#   filename: string '/path/to/file.png'
def plot_volume_orthoslices(vol, filename):
    """Writes an image file containing ortho-slices of the input volume. Generates same visualization as matlab function
    'tom_volxyz' from TOM toolbox.
    If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
    Else, it assumes that the volume is tomographic data, and plots in gray scale.

    Args:
        vol (3D numpy array)
        filename (str): '/path/to/file.png'
    """

    # Get central slices along each dimension:
    dim = vol.shape
    idx0 = np.int( np.round(dim[0]/2) )
    idx1 = np.int( np.round(dim[1]/2) )
    idx2 = np.int( np.round(dim[2]/2) )

    slice0 = vol[idx0,:,:]
    slice1 = vol[:,idx1,:]
    slice2 = vol[:,:,idx2]

    # Build image containing orthoslices:
    img_array = np.zeros((slice0.shape[0]+slice1.shape[0], slice0.shape[1]+slice1.shape[0]))
    img_array[0:slice0.shape[0], 0:slice0.shape[1]] = slice0
    img_array[slice0.shape[0]-1:-1, 0:slice0.shape[1]] = slice1
    img_array[0:slice0.shape[0], slice0.shape[1]-1:-1] = np.flipud(np.rot90(slice2))

    # Drop the plot:
    fig = plt.figure(figsize=(10,10))
    if vol.dtype==np.int8:
        plt.imshow(img_array, cmap='CMRmap', vmin=np.min(vol), vmax=np.max(vol))
    else:
        mu  = np.mean(vol) # Get mean and std of data for plot range:
        sig = np.std(vol)
        plt.imshow(img_array, cmap='gray', vmin=mu-5*sig, vmax=mu+5*sig)
    fig.savefig(filename)