import os
import pandas as pd
import numpy as np

headers = ['Tomo3D', 'Z', 'Y', 'X', 'Q1', 'Q2',	'Q3', 'Class']
base_dir = '/mnt/Data/Cryo-ET/data/simulated_data/'
output_dir = '/mnt/Data/Cryo-ET/data/simulated_data/'

csv_content = pd.read_csv(os.path.join(base_dir, 'tomo_rec_0_snr0.19.csv'))
content = np.asarray(csv_content.values)
ds_factor = 10
for row in range(len(content)):
    content[row][1] = content[row][1] / ds_factor
    content[row][2] = content[row][2] / ds_factor
    content[row][3] = content[row][3] / ds_factor

df = pd.DataFrame(content)
df.to_csv(os.path.join(output_dir, "tomo_rec_0_snr0.19_downsampled.csv"), index=False)








