import numpy as np
from lxml import etree
import pandas as pd
import os

tomoid = 9
# base_dir = "/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_" + str(tomoid) + "/"
# base_dir2 = "/mnt/Data/Cryo-ET/DeepET/data2/images/"


base_dir = "/mnt/Data/Cryo-ET/DeepET/data/SHREC/" + str(tomoid) + "/"
base_dir2 = "/mnt/Data/Cryo-ET/DeepET/data/SHREC/" + str(tomoid) + "/"

def read_xml2(filename, bdir, bdir2, tid):
    tree = etree.parse(filename)
    objl_xml = tree.getroot()

    obj_list = []
    column_names = ["name", "z", "y", "x", "phi", "the", "psi", "Class"]
    # content_pt = []
    # content_rb = []
    content = []
    for p in range(len(objl_xml)):
        lbl = objl_xml[p].get('class_label')
        name = str(bdir2) + str(tid) + "_resampled.mrc"
        z = objl_xml[p].get('z')
        y = objl_xml[p].get('y')
        x = objl_xml[p].get('x')
        phi = objl_xml[p].get('phi')
        the = objl_xml[p].get('the')
        psi = objl_xml[p].get('psi')
        # cls = ""
        if lbl == '12':
            cls = "4d8q"
            # content_pt.append([name, z, y, x, phi, the, psi, cls])
        # if lbl == '2':
        #     cls = "rb"
            # content_rb.append([name, z, y, x, phi, the, psi, cls])
            content.append([name, z, y, x, phi, the, psi, cls])

    df = pd.DataFrame(content, columns=column_names)
    csv = base_dir + '/T' + str(tomoid) + '_4d8q.csv'
    df.to_csv(csv, index=False)

    # df_pt = pd.DataFrame(content_pt, columns=column_names)
    # csv_pt = base_dir + '/T' + str(tomoid) + '_PT.csv'
    # df_pt.to_csv(csv_pt, index=False)
    #
    # df_rb = pd.DataFrame(content_rb, columns=column_names)
    # csv_rb = base_dir + '/T' + str(tomoid) + '_RB.csv'
    # df_rb.to_csv(csv_rb, index=False)

    return obj_list


object_list = read_xml2(os.path.join(base_dir, "objectlist" + str(tomoid) + ".xml"), base_dir, base_dir2, tomoid)
