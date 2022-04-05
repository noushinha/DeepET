import numpy as np
from lxml import etree
import pandas as pd
import os

tomoid = 2
def read_xml2(filename, tomoid):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    column_names = ["z", "y", "x", "phi", "the", "psi", "Class"]
    content = []
    particle_type = "1bxn_3gl1_4d8q"
    cnt = 0
    for p in range(len(objl_xml)):
        lbl = objl_xml[p].get('class_label')
        cls = ""
        z = objl_xml[p].get('z')
        y = objl_xml[p].get('y')
        x = objl_xml[p].get('x')
        phi = objl_xml[p].get('phi')
        the = objl_xml[p].get('the')
        psi = objl_xml[p].get('psi')
        if lbl == '1':
            cnt = cnt + 1
            cls = "1bxn"
        elif lbl == '8':
            cnt = cnt + 1
            cls = "3gl1"
        elif lbl == '12':
            cnt = cnt + 1
            cls = "4d8q"

        if cls != "":
            content.append([cls, z, y, x, phi, the, psi])
    print(cnt)
    df = pd.DataFrame(content, columns=column_names)
    # csv = base_dir + '/T' + str(tomoid) + '_PTRB.csv'
    # df.to_csv(csv, index=False)
    np.savetxt(r'/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(tomoid) + '/particle_locations_model_' + str(tomoid) + '_' + str(particle_type) + '.txt', df.values, fmt='%s')

    return obj_list


# base_dir = '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_8'
base_dir = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(tomoid) + '/'
object_list = read_xml2(os.path.join(base_dir, 'objectlist' + str(tomoid) + '.xml'), tomoid)

