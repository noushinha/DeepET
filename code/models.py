# ============================================================================================
# DeepET - a deep learning framework for segmentation and classification of
#                  macromolecules in Cryo Electron Tomograms (Cryo-ET)
# ============================================================================================
# Copyright (c) 2021 - now
# ZIB - Department of Visual and Data Centric
# Author: Noushin Hajarolasvadi
# Team Leader: Daniel Baum
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from tensorflow.keras import backend, callbacks, Model


# loading batches of data
class DataPreparation(keras.utils.Sequence):
    """we iterate over the data as numpy arrays"""

    def __init__(self, obj):
        self.batch_size = obj.batch_size
        self.patch_size = obj.patch_size
        self.img_size = obj.img_size
        self.input_img_paths = obj.input_img_paths
        self.target_img_paths = obj.target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        # Returns tuple (input, target) correspond to batch #idx
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            y[y == 255] = 1
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y


class CNNModels:
    def __init__(self, obj):
        datum = DataPreparation(obj)
        self.obj = obj
        dim = len(self.obj.img_size)
        self.width = self.obj.img_size[0]
        self.height = self.obj.img_size[1]

        if dim == 3:
            self.depth = self.obj.img_size[2]

        self.num_class = self.obj.num_class

    def unet2d(self):
        # The original 2D UNET mdoel
        # inputs = keras.Input(shape=self.width + (3,))
        input_img = layers.Input(shape=(self.width, self.height, 1))

        # down-sampling part of the network
        x = layers.Conv2D(32, 3, strides=2, padding="same")(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # up-sampling part of the network
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_class, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(input_img, outputs)
        return model

    def unet3d(self, obj):
        # The UNET model from DeepFinder
        input_img = layers.Input(shape=(self.width, self.height, self.depth, 1))

        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_img)
        high = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.MaxPooling3D((2, 2, 2), strides=None)(high)

        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
        mid = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.MaxPooling3D((2, 2, 2), strides=None)(mid)

        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        x = layers.Conv3D(64, (2, 2, 2), padding='same', activation='relu')(x)

        x = layers.concatenate([x, mid])
        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(48, (3, 3, 3), padding='same', activation='relu')(x)

        x = layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        x = layers.Conv3D(48, (2, 2, 2), padding='same', activation='relu')(x)

        x = layers.concatenate([x, high])
        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)

        output = layers.Conv3D(self.num_class, (1, 1, 1), padding='same', activation='softmax')(x)

        model = Model(input_img, output)
        return model
