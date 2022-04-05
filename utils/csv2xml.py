import numpy
import numpy as np
import pandas as pd
from lxml import etree
from collections import OrderedDict

reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
                                    "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})


def write_xml():
    for i in range(0, 10):
        objl_xml = etree.Element('objlist')
        xml_path = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(i) + '/objectlist' + str(i) + '.xml'
        csv_path = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(i) + '/particle_locations_model_' + str(i) + '.csv'
        data = pd.read_csv(csv_path)
        content = np.asarray(data.values)
        for row in range(len(content)):
            tidx = i
            objid = row
            lbl = reversed_class_names[content[row][7]]

            x = int(content[row][3])
            y = int(content[row][2])
            z = int(content[row][1])
            phi = content[row][6]
            psi = content[row][5]
            the = content[row][4]

            obj = etree.SubElement(objl_xml, 'object')

            if tidx is not None:
                obj.set('tomo_idx', str(tidx))
            if objid is not None:
                obj.set('obj_id', str(objid))

            obj.set('class_label', str(lbl))
            obj.set('x', '%d' % x)
            obj.set('y', '%d' % y)
            obj.set('z', '%d' % z)
            obj.set('phi', '%.3f' % phi)
            obj.set('psi', '%.3f' % psi)
            obj.set('the', '%.3f' % the)

            tree = etree.ElementTree(objl_xml)
            tree.write(xml_path, pretty_print=True)

write_xml()