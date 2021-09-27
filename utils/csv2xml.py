import numpy
import numpy as np
import pandas as pd
from lxml import etree


def write_xml(content, output_path):
    objl_xml = etree.Element('objlist')
    for row in range(len(content)):
        tidx = 8
        objid = row
        x = content[row][3]
        y = content[row][2]
        z = content[row][1]
        phi = content[row][4]
        psi = content[row][5]
        the = content[row][6]
        if content[row][7] == "pt":
            lbl = 1
        else:
            lbl = 2

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

data = pd.read_csv('/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_8/proteasome8.csv')
data = np.asarray(data.values)
write_xml(data, '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_8/objectlist8.xml')