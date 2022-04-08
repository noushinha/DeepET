import os
import numpy as np
import pandas as pd
from glob import glob
import re

header_list = ['Tomo3D', 'Z', 'Y', 'X', 'Q1', 'Q2',	'Q3', 'Q4', 'Class']
base_dir = '/mnt/Data/Cryo-ET/data/simulated_data/original_tomos/'
output_dir = '/mnt/Data/Cryo-ET/data/simulated_data/original_tomos/csv/'

list_csvs = glob(os.path.join(base_dir, "*.mrc"))
list_csvs.sort(key=lambda f: int(re.sub('\D', '', f)))

csv_content = pd.read_csv(os.path.join(base_dir, 'tomos_motif_list.csv'), sep=r'\t', engine="python")
content = np.asarray(csv_content.values)
csv_content = []
for f in range(len(list_csvs)):
    csv_content = []
    ith_filename = str.split(list_csvs[f], '/')
    ith_filename = ith_filename[-1]
    for row in range(len(content)):
        mrc_filename = str.split(content[row][3], '/')
        mrc_filename = mrc_filename[-1]
        cls = ''

        if mrc_filename == ith_filename:
            x = content[row][8]
            y = content[row][9]
            z = content[row][10]
            q1 = content[row][11]
            q2 = content[row][12]
            q3 = content[row][13]
            q4 = content[row][14]
            if content[row][6] == 'pdb_4v4r':
                cls = 'rb'
            elif content[row][6] == 'pdb_3j9i':
                cls = 'pt'
            file_lcoation = '/mnt/Data/Cryo-ET/data/simulated_data/downsampled_tomos/' + str.replace(mrc_filename, '0.', '')
            csv_content.append([file_lcoation, z, y, x, q1, q2, q3, q4, cls])


    new_mrc_filename = str.replace(ith_filename, '0.', '')
    csv_filename = str.split(new_mrc_filename, '.mrc')
    csv_filename = csv_filename[0] + ".csv"
    df = pd.DataFrame(csv_content)
    df.to_csv(os.path.join(output_dir, csv_filename), index=False, header=header_list)

