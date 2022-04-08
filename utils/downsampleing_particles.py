import os
import pandas as pd
import numpy as np
from glob import glob
import re


header_list = ['Tomo3D', 'Z', 'Y', 'X', 'Q1', 'Q2',	'Q3', 'Q4', 'Class']
base_dir = '/mnt/Data/Cryo-ET/data/simulated_data/original_tomos/csv'
output_dir = '/mnt/Data/Cryo-ET/data/simulated_data/downsampled_tomos/csv'

list_csvs = glob(os.path.join(base_dir, "*.csv"))
list_csvs.sort(key=lambda f: int(re.sub('\D', '', f)))

for f in range(len(list_csvs)):
    csv_filedir = list_csvs[f]
    csv_content = pd.read_csv(csv_filedir)
    content = np.asarray(csv_content.values)
    ds_factor = 10
    for row in range(len(content)):

        content[row][1] = content[row][1] / ds_factor
        content[row][2] = content[row][2] / ds_factor
        content[row][3] = content[row][3] / ds_factor

    df = pd.DataFrame(content)
    csv_filename = str.split(csv_filedir, '/')[-1]
    df.to_csv(os.path.join(output_dir, csv_filename), index=False, header=header_list)
