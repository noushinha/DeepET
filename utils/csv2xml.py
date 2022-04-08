import numpy
import numpy as np
import pandas as pd
from lxml import etree
from collections import OrderedDict

# reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
#                                     "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})

reversed_class_names = OrderedDict({"bg": 0, "pt": 1, "rb": 2})
def write_xml():
    # for i in range(0, 10):
    objl_xml = etree.Element('objlist')
    fname = 'tomo_rec_11_snr21'
    xml_path = '/mnt/Data/Cryo-ET/data/simulated_data/downsampled_tomos/csv/' + str(fname) + '.xml'
    csv_path = '/mnt/Data/Cryo-ET/data/simulated_data/downsampled_tomos/csv/' + str(fname) + '.csv'
    data = pd.read_csv(csv_path)
    content = np.asarray(data.values)
    for row in range(len(content)):
        tidx = 11
        objid = row
        lbl = reversed_class_names[content[row][8]]

        x = int(content[row][3])
        y = int(content[row][2])
        z = int(content[row][1])
        q1 = content[row][4]
        q2 = content[row][5]
        q3 = content[row][6]
        q4 = content[row][7]

        obj = etree.SubElement(objl_xml, 'object')

        if tidx is not None:
            obj.set('tomo_idx', str(tidx))
        if objid is not None:
            obj.set('obj_id', str(objid))

        obj.set('class_label', str(lbl))
        obj.set('x', '%d' % x)
        obj.set('y', '%d' % y)
        obj.set('z', '%d' % z)
        obj.set('Q1', '%.3f' % q1)
        obj.set('Q2', '%.3f' % q2)
        obj.set('Q3', '%.3f' % q3)
        obj.set('Q4', '%.3f' % q4)

        tree = etree.ElementTree(objl_xml)
        tree.write(xml_path, pretty_print=True)

write_xml()