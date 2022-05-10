import os
from radiomics import featureextractor
import SimpleITK as sitk
import re
import pandas as pd
from glob import glob
# import six
# import numpy as np
# import nibabel as nib
# import nrrd
# from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape

dataDir = '/mnt/Data/Cryo-ET/DeepET/data2/nifti/real'
paramPath = "/mnt/Data/Cryo-ET/DeepET/data2/nifti/params.yaml"

# list all images and masks
list_tr_tomos_IDs = glob(os.path.join(dataDir, "images/*.nrrd"))
list_tr_masks_IDs = glob(os.path.join(dataDir, "masks/*.nrrd"))
# sort images and masks
list_tr_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
list_tr_masks_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))
data = []
# for each pair of image and mask
for t in range(len(list_tr_tomos_IDs)):
    # read image and mask
    imagePath = os.path.join(dataDir, "images/patch_t" + str(t) + ".nrrd")
    maskPath = os.path.join(dataDir, "masks/patch_m" + str(t) + ".nrrd")
    image = sitk.ReadImage(imagePath)
    mask = sitk.ReadImage(maskPath)

    # ectract features
    print("Calculating features")
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    result = extractor.execute(imagePath, maskPath)

    print("Collecting ...")
    headers = []
    features = []
    # read and save all extracted features for this sample (sample is extracted particle patch from image and mask)
    for key, value in result.items():
        if key.find('diagnostics_') == -1:
            key.replace("original_", "")
            headers.append(key)
            features.append(value)
    print("********************************")
    data.append(features)

# save feature space as csv file
data_panda = pd.DataFrame(data, columns=headers)
data_panda.to_csv(os.path.join(dataDir, 'real_data_features.csv'), index=False)


    # print("\t", key, ":", value)
# settings = {'binWidth': 25,
#             'interpolator': sitk.sitkBSpline,
#             'resampledPixelSpacing': None}

# # Show Shape features
# shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
# shapeFeatures.enableAllFeatures()
#
# print('Calculating Shape features...')
# results = shapeFeatures.execute()
# print('done')
#
# print('Calculated Shape features: ')
# for (key, val) in six.iteritems(results):
#   print('  ', key, ':', val)

# # Show GLSZM features
# #
# glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
# glszmFeatures.enableAllFeatures()
#
# print('Calculating GLSZM features...')
# results = glszmFeatures.execute()
# print('done')
#
# print('Calculated GLSZM features: ')
# for (key, val) in six.iteritems(results):
#   print('  ', key, ':', val)


# Show GLCM features
# glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
# glcmFeatures.enableAllFeatures()
#
# print('Calculating GLCM features...')
# results = glcmFeatures.execute()
# print('done')
#
# print('Calculated GLCM features: ')
# for (key, val) in six.iteritems(results):
#   print('  ', key, ':', val)

# # Additonally, store the location of the example parameter file, stored in \pyradiomics\bin
# paramPath = os.path.join(dataDir, "Params.yaml")
# print("Parameter file, absolute path:", os.path.abspath(paramPath))
#
# # Instantiate the extractor
# extractor = featureextractor.RadiomicsFeatureExtractor()
#
# print('Extraction parameters:\n\t', extractor.settings)
# print('Enabled filters:\n\t', extractor.enabledImagetypes)
# print('Enabled features:\n\t', extractor.enabledFeatures)
#
# result = extractor.execute(imagePath, maskPath)
# print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
# print('')
# print('Calculated features')
# for key, value in six.iteritems(result):
#     print('\t', key, ':', value)
