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
import pandas as pd
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gui import theme_style
from lxml import etree

def display(message):
    print(message)


def is_list(listl, var):
    if type(listl) != list:
        theme_style.display_message('variable "' + var + '" is ' + str(type(listl)) + '. list is expected.')
        sys.exit()


def is_3D(arr, var):
    if type(arr) != np.ndarray or len(arr.shape) != 3:
        theme_style.display_message('variable "' + var + '" is ' + str(len(arr.shape)) + str(type(arr)) + '. 3D Numpy array is expected.')
        sys.exit()


def is_empty(arr, var):
   if arr.size == 0:
       theme_style.display_message('array "' + var + '" is empty. Non empty array is expected.')
       sys.exit()


def is_file(filename):
    from pathlib import Path
    if not Path(filename).is_file():
        theme_style.display_message('file "' + filename + '" does not exist. A valid file is required.')
        sys.exit()
    else:
        return 1


def is_dir(dirpath):
    if not os.path.isdir(dirpath):
        # display('path "' + dirpath + '" does not exist. A valid directory is required.')
        theme_style.display_message('path "' + dirpath + '" does not exist. A valid directory is required.')
        sys.exit()
    else:
        return 1


def is_int(num, var):
    if type(num)!=int and type(num)!=np.int8 and type(num)!=np.int16:
        theme_style.display_message('variable "' + var + '" is ' + str(type(num)) + '. An integer is required.')
        sys.exit()


def is_positive(num, var):
    is_int(num, var)
    if num <= 0:
        theme_style.display_message('variable "'+var+'" is negative. positive value is required.')
        sys.exit()


def is_same_shape(num1, num2):
    if num1 != num2:
        theme_style.display_message('the image and the target mask are not of same size!')
        sys.exit()


# def is_factor4(self, num, var):
#     self.is_int(num, var)
#     if num % 4 != 0:
#         display('variable "' + var + '" should be a multiple of 4.')
#         sys.exit()
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


def throwErr(errtype='None'):
    errval = errtype.split(":")
    if errval[0] == 'ext':
        print('The file extension {extension} is not supported.'.format(extension=errval[1]))


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
        if tomo_idx is not None:
            tomo_idx = int(tomo_idx)
        add_obj(obj_list, tomo_idx=tomo_idx, obj_id=object_id, label=int(lbl), coord=(float(z), float(y), float(x)))
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

    if (x > tomodim[2] - p_in):
        x = tomodim[2] - p_in
    if (y > tomodim[1] - p_in):
        y = tomodim[1] - p_in
    if (z > tomodim[0] - p_in):
        z = tomodim[0] - p_in

    return x, y, z

def correct_center_positions(xc, yc, zc, dim, offset):
    # If there are still few pixels at the end:
    if xc[-1] < dim[2] - offset:
        xc = xc + [dim[2] - offset, ]
    if yc[-1] < dim[1] - offset:
        yc = yc + [dim[1] - offset, ]
    if zc[-1] < dim[0] - offset:
        zc = zc + [dim[0] - offset, ]

    return xc, yc, zc

def add_obj(obj_list, label, coord, obj_id=None, tomo_idx=None, c_size=None):
    obj = {
        'tomo_idx': tomo_idx,
        'obj_id': obj_id ,
        'label': label,
        'x': coord[2],
        'y': coord[1],
        'z': coord[0],
        'c_size': c_size
    }

    obj_list.append(obj)
    return obj_list

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
    if is_file(filename):
        import mrcfile as mrc
        with mrc.open(filename, mode='r+', permissive=True) as mc:
            mc.update_header_from_data()
            mrc_tomo = mc.data
        is_empty(mrc_tomo, 'mrc_tomo')
    return mrc_tomo


def write_mrc(array, filename):
    """ This function writes an mrc file
        Args: filename: /saving/path
              array: nd array
    """
    import mrcfile as mrc
    with mrc.new(filename, overwrite=True) as mc:
        mc.set_data(array)


def write_xml(objlist, output_path):
    objl_xml = etree.Element('objlist')
    for i in range(len(objlist)):
        tidx = objlist[i]['tomo_idx']
        objid = objlist[i]['obj_id']
        lbl = objlist[i]['label']
        x = objlist[i]['x']
        y = objlist[i]['y']
        z = objlist[i]['z']
        csize = objlist[i]['c_size']

        obj = etree.SubElement(objl_xml, 'object')

        if tidx is not None:
            obj.set('tomo_idx', str(tidx))
        if objid is not None:
            obj.set('obj_id', str(objid))

        obj.set('class_label', str(lbl))
        obj.set('x', '%.3f' % x)
        obj.set('y', '%.3f' % y)
        obj.set('z', '%.3f' % z)
        if csize is not None:
            obj.set('cluster_size', str(csize))

    tree = etree.ElementTree(objl_xml)
    tree.write(output_path, pretty_print=True)


def get_coords(dim):
    """ This function receives dimensions of a 3D volume and returns center coords.
        Args: dim: the 3D volume shape(tomogram - e.g an mrc file)
        Returns: center coordination in x, y, z
    """
    x = np.int(np.round(dim[1] / 2))
    y = np.int(np.round(dim[2] / 2))
    z = np.int(np.round(dim[0] / 2))
    return x, y, z


def get_planes(vol):
    """ This function receives a 3D volume and returns orthoclices of xy, zx, and zy planes.
        Args: vol : the 3D volume (tomogram - e.g an mrc file)
              coords : look get_coords (coords[0] -> z, coords[1] -> y, coords[2] -> x)
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
    ########## or ##########
    # for i in range(class_num):
    #     boxcolor.append(np.random.randint(low=0, high=255))
    # NOPE, none worked better than the following one
    boxcolor = np.random.permutation(255)[:class_num]
    return boxcolor


def radius_list(class_name, str_radiuslist):
    radiuslist = str_radiuslist.split(",")
    class_radlist = dict.fromkeys(class_name, 0)
    cnt = 0
    for i in class_name:
        class_radlist[i] = int(radiuslist[cnt])
        cnt = cnt + 1

    return class_radlist


def generate_spheres(content, target_mask, class_radilist):
    """ preprocessing and preparing lists of ref. spheres for generating masks
    """
    # radi_ref: reference list for radius of classes (spheres)
    # The references order should be the same as order of class labels.
    # e.g: 1st list item -> reference for label 1;
    # 2nd list item -> reference for label 2; etc.
    radius_vals = list(class_radilist.values())
    Rmax = max(radius_vals)
    radi_ref = []
    for idx in range(len(radius_vals)):
        sphere = prepare_ref_sphere(radius_vals[idx])
        radi_ref.append(sphere)
    target_array = generate_masks(content, target_mask, radi_ref, class_radilist)
    return target_array


def generate_masks(content, target_mask, radi_ref, class_radilist):
    """ Having a annotation list, this function generates the segmentation mask
    Args: content: Should contain the following data and in order:
                   [Path/to/image/file,z,y,x,[...optional columns],label]
        target_mask: all zero 3D numpy array to initializes the mask target.
                     dimension order should be [z,y,x] to be compatible with content
    Returns: 3D np array of the mask, '0' is taken for background class, '1','2',... for rest of classes.
    """
    is_3D(target_mask, 'target_mask')
    is_list(radi_ref, 'radi_ref')

    boxcolor = color_pallete(len(class_radilist.keys()), list(class_radilist.keys()))

    ann_num = content.shape[0]  # number of available annotation we have
    dim = target_mask.shape  # image shape (mask shape is same as image shape)
    # for each annotation
    for row in range(ann_num):
        cls_ann = int(tuple(class_radilist.keys()).index(content[row][-1]))
        z = int(content[row][1])
        y = int(content[row][2])
        x = int(content[row][3])
        display('Annotating point ' + str(row + 1) + ' / ' + str(ann_num) +
                ' with class ' + str(content[row][-1]) +
                ' and color ' + str(boxcolor[cls_ann]))

        ref = radi_ref[cls_ann - 1]
        cOffset = np.int(np.floor(ref.shape[0] / 2))  # guarantees a cubic reference

        # identify coordinates of particle in mask
        obj_voxels = np.nonzero(ref == 1)
        x_coord = obj_voxels[2] + x - cOffset
        y_coord = obj_voxels[1] + y - cOffset
        z_coord = obj_voxels[0] + z - cOffset

        for idx in range(x_coord.size):
            xVox = x_coord[idx]
            yVox = y_coord[idx]
            zVox = z_coord[idx]
            # check that after offset transfer the coords are in the boudnary of tomo
            if xVox >= 0 and xVox < dim[2] and yVox >= 0 and yVox < dim[1] and zVox >= 0 and zVox < dim[0]:
                target_mask[zVox, yVox, xVox] = cls_ann+1  # boxcolor[cls_ann]
    return target_mask


def prepare_ref_sphere(radi):
    """ This function creates a sphere for the radius it receives
        Args: radi: list of particle radius
              dim: list of x, y, z radius of the sphere
        Returns: a sphere
    """
    dim = [2 * radi, 2 * radi, 2 * radi]  # not necessary but makes the code legible
    center = np.floor((dim[0]/2, dim[1]/2, dim[2]/2))
    x, y, z = np.meshgrid(range(dim[0]), range(dim[1]), range(dim[2]))

    Sph = ((x - center[0])/radi)**2 + ((y - center[1])/radi)**2 + ((z - center[2])/radi)**2
    Sph = np.int8(Sph <= 1)
    return Sph


def save_volume(input_array, filename):
    """saves a png image from the generated masks if int8 type: a labelmap is saved in color scale.
       otherwise, saves in gray scale.
    Args: vol (3D numpy array)
          filename (str): '/path/to/file'
    returns: image file
    """

    # Get central slices along each dimension:
    dim = input_array.shape
    z = np.int(np.round(dim[0]/2))
    y = np.int(np.round(dim[1]/2))
    x = np.int(np.round(dim[2]/2))

    slice0 = input_array[z, :, :]
    slice1 = input_array[:, y, :]
    slice2 = input_array[:, :, x]

    # creating an image out of ortho-slices:
    input_img = np.zeros((slice0.shape[0]+slice1.shape[0], slice0.shape[1]+slice1.shape[0]))
    input_img[0:slice0.shape[0], 0:slice0.shape[1]] = slice0
    input_img[slice0.shape[0]-1:-1, 0:slice0.shape[1]] = slice1
    input_img[0:slice0.shape[0], slice0.shape[1]-1:-1] = np.flipud(np.rot90(slice2))

    # plot and save:
    fig = plt.figure(figsize=(10, 10))
    if input_array.dtype == np.int8:
        plt.imshow(input_img, cmap='CMRmap', vmin=np.min(input_array), vmax=np.max(input_array))
    else:
        # calculating mean and std of data
        mu = np.mean(input_array)
        sig = np.std(input_array)
        plt.imshow(input_img, cmap='gray', vmin=mu-5*sig, vmax=mu+5*sig)
    # plt.show()
    fig.savefig(filename)


# saving labels or predicted probablities as a npy file
def save_npy(data, output_path, flag="Train", name="Probabilities"):
    np.save(os.path.join(output_path, flag + "_" + name + ".npy"), data)


# saving labels or predicted probablities as a csv file
def save_csv(data, output_path, flag="Train", name="Probabilities"):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, flag + "_" + name + ".csv"))
