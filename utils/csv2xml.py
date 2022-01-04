import numpy
import numpy as np
import pandas as pd
from lxml import etree
from collections import OrderedDict

reversed_class_names = OrderedDict({"bg": 0, "1bxn": 1, "1qvr": 2, "1s3x": 3, "1u6g": 4, "2cg9": 5, "3cf3": 6,
                                    "3d2f": 7, "3gl1": 8, "3h84": 9, "3qm1": 10, "4b4t": 11, "4d8q": 12})
def write_xml(content, output_path, tomo_id):
    objl_xml = etree.Element('objlist')
    # voxSize = 4.537897311
    for row in range(len(content)):
        tidx = tomo_id
        objid = row
        # x = int(content[row][3] / voxSize) - 1  # content[row][3]
        # y = int(content[row][2] / voxSize) - 1  # content[row][2]
        # z = int(content[row][1] / voxSize) - 1  # content[row][1]
        x = int(content[row][3])
        y = int(content[row][2])
        z = int(content[row][1])
        phi = content[row][6]
        psi = content[row][5]
        the = content[row][4]
        # if content[row][7] == "pt":
        #     lbl = 1
        # else:
        #     lbl = 2
        lbl = reversed_class_names[content[row][7]]
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
    tree.write(output_path, pretty_print=True)


tomo_id = 9
csv_path = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(tomo_id) + '/particle_locations_model_' + str(tomo_id) + '.csv'
data = pd.read_csv(csv_path)
data = np.asarray(data.values)
xml_path = '/mnt/Data/Cryo-ET/DeepET/data/SHREC/' + str(tomo_id) + '/objectlist' + str(tomo_id) + '.xml'
write_xml(data, xml_path, tomo_id)
