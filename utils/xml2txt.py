import numpy as np
from lxml import etree
import pandas as pd
import os


def read_xml2(filename):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    column_names = ["z", "y", "x", "phi", "the", "psi", "Class"]
    content = []

    for p in range(len(objl_xml)):
        lbl = objl_xml[p].get('class_label')

        z = objl_xml[p].get('z')
        y = objl_xml[p].get('y')
        x = objl_xml[p].get('x')
        phi = objl_xml[p].get('phi')
        the = objl_xml[p].get('the')
        psi = objl_xml[p].get('psi')
        if lbl == '1':
            cls = "pt"
            # content_pt.append([name, z, y, x, phi, the, psi, cls])
        if lbl == '2':
            cls = "rb"
            # content_rb.append([name, z, y, x, phi, the, psi, cls])
        # content.append([name, z, y, x, phi, the, psi, cls])
        content.append([cls, z, y, x, phi, the, psi])
    df = pd.DataFrame(content, columns=column_names)
    # csv = base_dir + '/T' + str(tomoid) + '_PTRB.csv'
    # df.to_csv(csv, index=False)
    np.savetxt(r'/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_8/particle_locations_model.txt', df.values, fmt='%s')

    return obj_list


base_dir = '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_8'
object_list = read_xml2(os.path.join(base_dir, 'objectlist8.xml'))

