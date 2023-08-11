import os
import sys
import time
from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import keras
from keras import Model, layers, optimizers, metrics
from keras.utils import to_categorical
import keras.backend as k

from utils import utility_tools as ut
from utils.losses import *
from sklearn.metrics import precision_recall_fscore_support

# defining initial variables
base_path = '/media/noushin/Data/Cryo-ET/DeepET/data2/'
output_path = '/media/noushin/Data/Cryo-ET/DeepET/data2/results/pretraining/real_vs_synthetic/'
real_img_paths = os.path.join(base_path, 'images')
synt_img_paths = os.path.join(base_path, 'Artificial/V2/combined/')

# training hyper parameters
epochs = 50
learning_rate = 0.0001
batch_size = 64
vald_prc = 0.125
patch_size = 64
classNum = 2

real_valid_samples = []
synt_valid_samples = []
real_train_samples = []
synt_train_samples = []

# loading data set
def fetch_dataset():
    print("fetching dataset...")
    list_real_tomos_IDs = glob(os.path.join(real_img_paths, "*_resampled.mrc"))
    list_synt_tomos_IDs = glob(os.path.join(synt_img_paths, "*_resampled.mrc"))

    list_real_tomos_IDs.sort(key=lambda f: int(re.sub('\D', '', f)))
    list_synt_tomos_IDs.sort(key=lambda r: int(re.sub('\D', '', r)))

    return list_real_tomos_IDs, list_synt_tomos_IDs


def fetch_tomos(list_real_imgs, list_synt_imgs):
    tomograms = []
    print("fetching tomograms...")
    start = time.perf_counter()
    for t in range(len(list_real_imgs)):
        print("********** Fetch Real Tomogram {tnum} **********".format(tnum=list_real_imgs[t]))
        real_tomo = ut.read_mrc(list_real_imgs[t])
        tomograms.append(real_tomo)

    real_annotation_list = ut.read_xml2(os.path.join(real_img_paths, "object_list_train.xml"))
    np.random.shuffle(real_annotation_list)
    real_annotation_list = real_annotation_list[0:4228]
    print("Real Data annotations: {lenreal}".format(lenreal=len(real_annotation_list)))

    for t in range(len(list_synt_imgs)):
        print("********** Fetch Tomogram {tnum} **********".format(tnum=list_synt_imgs[t]))
        synt_tomo = ut.read_mrc(list_synt_imgs[t])
        tomograms.append(synt_tomo)

    synt_annotation_list = ut.read_xml2(os.path.join(synt_img_paths, "object_list_train.xml"))
    np.random.shuffle(synt_annotation_list)
    synt_annotation_list = synt_annotation_list[0:4228]
    print("Synthetic Data annotations: {lensynt}".format(lensynt=len(synt_annotation_list)))

    end = time.perf_counter()
    process_time = (end - start)
    print("tomograms fetched in {:.2f} seconds.".format(round(process_time, 2)))

    return tomograms, real_annotation_list, synt_annotation_list


def fetch_batch(tomograms, real_annotation_list, synt_annotation_list,
                b, bsize, flag_new_epoch, flag_new_batch):
    global real_train_samples
    global synt_train_samples
    global real_valid_samples
    global synt_valid_samples

    half_batch = int(batch_size / 2)
    bstart = b * batch_size
    bend = (b * batch_size) + batch_size
    if b == 0:
        print("********** " + flag_new_batch + " **********")

    mid_dim = int(np.floor(patch_size / 2))

    num_real_train_samples = int(np.round(len(real_annotation_list) * (1-vald_prc)))
    num_synt_train_samples = int(np.round(len(synt_annotation_list) * (1 - vald_prc)))

    if flag_new_epoch:
        # shuffle list of all samples so in the new epoch we get different train and valid samples
        np.random.shuffle(real_annotation_list)
        np.random.shuffle(synt_annotation_list)

        real_train_samples = real_annotation_list[0:num_real_train_samples]
        real_valid_samples = real_annotation_list[num_real_train_samples:-1]
        synt_train_samples = synt_annotation_list[0:num_synt_train_samples]
        synt_valid_samples = synt_annotation_list[num_synt_train_samples:-1]

    batch_tomo = np.zeros((bsize, patch_size, patch_size, patch_size, 1))


    cnt = 0
    tomo_idx = 0
    cnt_real = 0
    cnt_synt = 0
    for i in range(bstart, bend):
        if flag_new_batch == "Train":
            if cnt_real < half_batch:
                tomo_idx = int(real_train_samples[i]['tomo_idx'])
                tomo_info = real_train_samples[i]
                tomo = tomograms[tomo_idx]
                cnt_real += 1
            else:
                tomo_idx = int(synt_train_samples[i]['tomo_idx'])
                tomo_info = synt_train_samples[i]
                tomo = tomograms[int(tomo_idx+len(tomograms)/2)]
                cnt_synt += 1
        else:
            if cnt_real < half_batch:
                tomo_idx = int(real_valid_samples[i]['tomo_idx'])
                tomo_info = real_train_samples[i]
                tomo = tomograms[tomo_idx]
                cnt_real += 1
            else:
                tomo_idx = int(synt_valid_samples[i]['tomo_idx'])
                tomo_info = synt_train_samples[i]
                tomo = tomograms[int(tomo_idx+len(tomograms)/2)]
                cnt_synt += 1

        # Get patch position:
        x, y, z = ut.get_patch_position(tomo.shape, mid_dim, tomo_info, 0)

        # extract the patch:
        patch_tomo = tomo[z - mid_dim:z + mid_dim, y - mid_dim:y + mid_dim, x - mid_dim:x + mid_dim]
        patch_tomo = (patch_tomo - np.mean(patch_tomo)) / np.std(patch_tomo)



        # convert to categorical labels
        batch_tomo[cnt, :, :, :, 0] = patch_tomo
        cnt = cnt + 1

    # print(cnt_real)
    # print(cnt_synt)
    real_tomo_labels = np.array([0 for _ in range(half_batch)])
    synt_tomo_labels = np.array([1 for _ in range(half_batch)])

    batch_mask = np.concatenate((real_tomo_labels, synt_tomo_labels), axis=0)
    batch_mask = to_categorical(batch_mask)
    # print("batch of tomgorams shape is: {btomosh}".format(btomosh=batch_tomo.shape))
    # print("batch of label shape is: {blblsh}".format(blblsh=batch_mask.shape))

    batch_tomo, batch_mask = shuffle(batch_tomo, batch_mask)
    return batch_tomo, batch_mask


def unet_encoder(input_shape):

    # # The UNET model from DeepFinder
    input_img = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1))

    x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_img)
    # x = layers.Lambda(dropout(x))
    high = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

    x = layers.MaxPooling3D((2, 2, 2), strides=None)(high)

    x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
    mid = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

    x = layers.MaxPooling3D((2, 2, 2), strides=None)(mid)

    x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(.25)(x)
    # x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Flatten()(x)
    output = layers.Dense(units=2, activation="sigmoid")(x)

    model = Model(input_img, output)
    return model


real_tomo_IDs, synt_tomo_IDs = fetch_dataset()
tomos, real_particles, synt_particles = fetch_tomos(real_tomo_IDs, synt_tomo_IDs)
dataset_len = len(real_particles) + len(synt_particles)
print("dataset length = ", dataset_len)
# fitting the model
encoder = unet_encoder((patch_size, patch_size, patch_size))
encoder.summary()
optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)
# optimizer = optimizers.SGD(learning_rate=learning_rate, decay=0.0, momentum=0.9, nesterov=True)
encoder_metrics = [metrics.BinaryAccuracy(name='acc'), metrics.Precision(name='precision'),
                   metrics.Recall(name='recall'), metrics.AUC(name='auc')]
s = Semantic_loss_functions()

encoder.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
# encoder.compile(optimizer=optimizer, loss=s.bce_dice_loss, metrics=['accuracy'])

steps_per_epoch =  int(np.round((1-vald_prc) * len(real_particles)) / batch_size)
vald_steps_per_epoch = int(np.round(vald_prc * len(real_particles)) / batch_size) - 1

# Full training
label_list = []
for l_list in range(classNum):
    label_list.append(l_list)

for e in range(epochs):
    flag_new_epoch = True
    print("########## New Epoch ##########\n")
    k.set_value(encoder.optimizer.lr, learning_rate)
    for b in range(steps_per_epoch):
        # fetch the current batch of patches
        if b != 0:
            flag_new_epoch = False
        batch_tomo, batch_mask = fetch_batch(tomos, real_particles, synt_particles,
                                             b, batch_size, flag_new_epoch, "Train")
        loss_train = encoder.train_on_batch(batch_tomo, batch_mask)
        print('epoch %d/%d - b %d/%d - loss: %.3f - acc: %.3f - lr: %.7f' % (e + 1, epochs,
                                                                             b + 1, steps_per_epoch,
                                                                             loss_train[0], loss_train[1],
                                                                             k.eval(encoder.optimizer.lr)))
    for d in range(vald_steps_per_epoch):
        batch_tomo_vald, batch_mask_vald = fetch_batch(tomos, real_particles, synt_particles,
                                                       d, batch_size, False, "Validation")
        loss_val = encoder.evaluate(batch_tomo_vald, batch_mask_vald, verbose=0)
        batch_pred = encoder.predict(batch_tomo_vald)
        # scores = precision_recall_fscore_support(batch_mask_vald.argmax(axis=-1).flatten(),
        #                                          batch_pred.argmax(axis=-1).flatten(), average=None,
        #                                          labels=label_list, zero_division=0)
        print("val. loss: {vl}, val acc: {va}".format(vl=np.round(loss_val[0], 2),
                                                      va=np.round(loss_val[1], 2)))
        # print("F1 Score : {f1s}, \n Recall: {res}, \n Precision: {prs}".format(f1s=np.round(scores[2], 2),
        #                                                                        res=np.round(scores[1], 2),
        #                                                                        prs=np.round(scores[0], 2)))
        # print(np.unique(np.argmax(batch_pred, 4)))

encoder.save(os.path.join(output_path + '/real_synthetic_pretraining_weights.h5'))
