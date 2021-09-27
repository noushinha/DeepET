import numpy
import starfile
import pandas as pd


tomo_id = 23


def read_starfile(filename1, filename2):
    content1 = starfile.open(filename1)
    content2 = starfile.open(filename2)
    content1 = content1.to_numpy()
    content2 = content2.to_numpy()
    content1[:, 15] = "pt"
    content2[:, 15] = "rb"
    data_numpy = numpy.concatenate((content1, content2))
    return loaddata(data_numpy)


def loaddata(content):
    rows = content.shape[0]
    data = []
    for row_number in range(rows):
        x = content[row_number][5]
        y = content[row_number][6]
        z = content[row_number][7]
        phi = content[row_number][12]
        psi = content[row_number][13]
        the = content[row_number][14]
        cls = content[row_number][15]
        name = "data/invitro_RibosomeAndProteasome/tomo_" + str(tomo_id) + "/" + str(tomo_id) + ".mrc"
        cur_row = [str(name), z, y, x, phi, the, psi, cls]
        data.append(cur_row)
    return data

filename1 = '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_' + str(tomo_id) + '/proteasome' + str(tomo_id) + '.star'
filename2 = '/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_' + str(tomo_id) + '/ribosome' + str(tomo_id) + '.star'
data_numpy = read_starfile(filename1, filename2)

data_panda = pd.DataFrame(data_numpy, columns=['name', 'z', 'y', 'x', 'phi', 'psi', 'the', 'Class'])
data_panda.to_csv('/mnt/Data/Cryo-ET/DeepET/data/invitro_RibosomeAndProteasome/tomo_' + str(tomo_id) + '/proteasome' + str(tomo_id) + '.csv', index=False)
