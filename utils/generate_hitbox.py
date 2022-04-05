import os
import mrcfile as mrc
import numpy as np
from lxml import etree
from params import *
from random import randrange


def read_mrc(filename):
    with mrc.open(filename, mode='r+', permissive=True) as mc:
        mc.update_header_from_data()
        mrc_tomo = mc.data
    print(mc.header)
    return mrc_tomo


def write_mrc(array, filename):
    with mrc.new(filename, overwrite=True) as mc:
        mc.set_data(array)
        # mc.update_header_from_data()
        # print("########## 111111 ############")
        # mc.print_header()
        # vox_sizes = mc.voxel_size.copy()
        # vox_sizes.flags.writeable = True
        # vox_sizes = (2, 2, 2)
        # mc.voxel_size = vox_sizes
        # mc.header.nx = mc.data.shape[-1]
        # mc.header.exttyp = 'FEI1'
        # mc.set_extended_header(mc.header)

    read_mrc(filename)


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


tomo_id = 9
particle_type = "4d8q"
base_dir = "/mnt/Data/Cryo-ET/DeepET/data/SHREC/" + str(tomo_id) + "/"
output_dir = "/mnt/Data/Cryo-ET/DeepET/data/SHREC/" + str(tomo_id) + "/"
tomo_name = base_dir + 'micrographs/reconstruction_model_0' + str(tomo_id) + '.mrc'
tomo = read_mrc(tomo_name)
hitbox_size = tomo - tomo
print(tomo.shape)

# you need xml files listing particles with following info:
# <object tomo_idx="23" obj_id="0" class_label="1" x="209" y="243" z="106" phi="113.678" psi="85.385" the="232.794"/>
# also name of file must be objectlist<tomo_id>>_<particle_type>.xml
list_annotations = read_xml2(os.path.join(output_dir, "objectlist" + str(tomo_id) + ".xml"))
# class_names_list = ["12"]  # 4D8Q
# class_names_list = ["8"]  # 3GL1
# class_names_list = ["1"]  # 1BXN

my_class_radius = [0, 6, 6, 3, 6, 6, 7, 6, 4, 4, 3, 10, 8]
# for each annotation
cnt = 0
for row in range(0, len(list_annotations)):
    lbl = int(list_annotations[row]['label'])
    # if lbl == 12 or lbl == 8 or lbl == 1:
    if lbl == 12:
        z = int(list_annotations[row]['z'])
        y = int(list_annotations[row]['y'])
        x = int(list_annotations[row]['x'])

        radi = my_class_radius[lbl]
        hitbox_size[z-radi:z+radi, y-radi:y+radi, x-radi:x+radi] = cnt  # 128.0
        cnt = cnt + 1


print(cnt)
print(np.unique(hitbox_size))
write_mrc(np.float32(hitbox_size), os.path.join(output_dir, "hitbox_" + str(tomo_id) + "_" + particle_type + ".mrc"))
