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

import time
import math
import shutil
import random
import string
from utils.plots import *
from utils.utility_tools import *
from utils.CyclicLR.clr_callback import CyclicLR
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.optimizers import *
from tensorflow.keras import Model  # backend, callbacks,
from tensorflow.keras import layers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img


# loading batches of data
class DataPreparation(keras.utils.Sequence):
    """we iterate over the data as numpy arrays"""

    def __init__(self, obj):
        self.batch_size = obj.batch_size
        self.patch_size = obj.patch_size
        self.img_size = obj.img_dim
        self.train_img_path = os.path.join(obj.base_path, "images/train/")
        self.train_target_path = os.path.join(obj.base_path, "targets/train/")
        self.test_img_path = os.path.join(obj.base_path, "images/test/")
        self.test_target_path = os.path.join(obj.base_path, "targets/test/")

        seednum = 2
        np.random.seed(seednum)
        random_str = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(16)])
        self.output_path = os.path.join(obj.output_path, random_str)
        os.makedirs(self.output_path)

        # check values
        is_positive(self.batch_size, 'batch_size')
        is_positive(self.patch_size, 'patch_size')
        is_dir(self.train_img_path)
        is_dir(self.train_target_path)
        is_dir(self.test_img_path)
        is_dir(self.test_target_path)
        is_dir(self.output_path)

    def __len__(self):
        return len(self.train_target_path) // self.batch_size

    def __getitem__(self, idx):
        # Returns tuple (input, target) correspond to batch #idx
        i = idx * self.batch_size
        batch_img_path = self.train_img_path[i: i + self.batch_size]
        batch_target_path = self.train_target_path[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_img_path):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_path):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # color_num = np.unique(img)  # number of classes
            # y[y == 255] = 1
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
        return x, y


class CNNModels:

    def __init__(self, obj):
        self.data = DataPreparation(obj)

        # define values
        self.lrr = []
        self.lrT = None
        self.net = None
        self.history = None
        self.optimizer = None
        self.checkpoint = None
        self.layer_name = None
        self.process_time = None

        # initialize values
        self.obj = obj
        self.lr = obj.lr
        self.opt = obj.opt
        self.loss = obj.loss
        self.epochs = obj.epochs
        self.metrics = obj.metrics
        self.model_type = obj.model
        self.num_class = self.obj.classNum
        self.width = self.obj.img_dim[0]
        self.height = self.obj.img_dim[1]
        if len(self.obj.img_dim) > 2:
            self.depth = self.obj.img_dim[2]

        # check values
        is_positive(self.epochs, 'epochs')
        is_positive(self.num_class, 'num_class')

        # initialize model
        self.train_model()

    def train_model(self):
        """
        This function starts the training procedure by calling
        different built-in functions of the class CNNModel
        """
        self.get_model()
        self.fit_model()
        self.plots()
        self.save()
        plt.show(block=True)

    def set_optimizer(self):
        self.optimizer = Adam(lr=self.lr, beta_1=.9, beta_2=.999, epsilon=1e-08, decay=0.0)

        if self.opt == "SGD":
            self.optimizer = SGD(lr=self.lr, decay=0.0, momentum=0.9, nesterov=True)
        elif self.opt == "Adagrad":
            self.optimizer = Adagrad(lr=self.lr, epsilon=1e-08, decay=0.0)
        elif self.opt == "Adadelta":
            self.optimizer = Adadelta(lr=self.lr, rho=0.95, epsilon=1e-08, decay=0.0)
        elif self.opt == "Adamax":
            self.optimizer = Adamax(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.opt == "Nadam":
            self.optimizer = Nadam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.opt == "RMSprop":
            self.optimizer = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)

    # learning rate schedule
    def step_decay(self):
        initial_lrate = self.lr
        drop = 0.5
        epochs_drop = 50
        lrate = initial_lrate * math.pow(drop, math.floor((1 + self.epochs) / epochs_drop))
        self.lrr.append(lrate)
        return lrate

    def set_lr(self):
        # learning schedule callback
        self.lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10,
                                    min_lr=1e-06, mode='min', verbose=1)
        if self.lrT == "step_decay":
            self.lr = LearningRateScheduler(self.step_decay())
        elif self.lrT == "lambda":
            self.lr = LearningRateScheduler(lambda this_epoch: self.lr * 0.99 ** this_epoch)
        elif self.lrT == "cyclic":
            self.lr = CyclicLR(base_lr=self.lr, max_lr=6e-04, step_size=500., mode='exp_range', gamma=0.99994)

    def get_lr_metric(self):
        def lr(y_true, y_pred):
            # lrr.append(float(K.get_value(optimizer.lr)))
            self.lr = self.optimizer.lr

    def set_compile(self):
        self.get_lr_metric()
        self.metrics = ['accuracy', self.lr]
        if self.loss != "tversky":
            self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])
        else:
            self.net.compile(optimizer=self.optimizer, loss=self.tversky_loss, metrics=[self.metrics])

    def set_checkpoint(self):

        self.checkpoint = ModelCheckpoint(self.data.output_path, monitor='val_acc',
                                          verbose=1, save_best_only=True, mode='max')

    def set_callback(self):
        callbacks_list = [self.checkpoint, self.lr]
        self.callbacks = "[checkpoint, reduce_lr]"
        # self.callbacks = [callbacks.ModelCheckpoint("weights.h5", save_best_only=True)]

    def get_model(self):
        if self.model_type == "2D UNet":
            self.net = self.unet2d()
        elif self.model_type == "3D UNet":
            self.net = self.unet3d()

        # set the properties of the mdoel
        self.set_optimizer()
        self.get_lr_metric()
        self.set_compile()
        print(self.net.summary())

    def fit_model(self):
        start = time.clock()
        # self.history = self.net.fit(self.train_data[train], self.train_labels_one_hot_coded,
        #                     epochs=self.epoch, batch_size=self.data.batch_size, shuffle=True,
        #                     validation_data=(self.train_data[vald], self.vald_labels_one_hot_coded),
        #                     callbacks=self.callbacks_list)
        end = time.clock()
        self.process_time = (end - start)

    def plots(self):
        start_point = 10 # dropping the first few point in plots due to unstable behavior of model
        cnf_matrix = np.zeros(shape=[self.num_class, self.num_class])
        # figure(num=1, figsize=(8, 6), dpi=80)
        # plot_folds_accuracy(model_history)

        # plt.figure(num=2, figsize=(8, 6), dpi=80)
        # plot_folds_loss(model_history)

        plt.figure(num=1, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.train_loss[start_point:], self.vald_loss[start_point:], isLoss=True)

        plt.figure(num=2, figsize=(8, 6), dpi=100)
        plot_train_vs_vald(self.train_acc[start_point:], self.vald_acc[start_point:])

        plt.figure(num=3, figsize=(8, 6), dpi=100)
        plot_lr(self.train_lr[start_point:])

        # Plot all ROC curves
        plt.figure(num=4, figsize=(8, 6), dpi=100)
        # plot_roc(test_labels_one_hot_coded, test_predicted_probs)

        # Compute confusion matrix
        # cnf_matrix = confusion_matrix(test_labels, test_predicted_labels)
        np.set_printoptions(precision=2)

        cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(np.average(cnf_matrix2.diagonal()))

        # Plot and save non-normalized confusion matrix
        plt.figure(num=5, figsize=(5, 5), dpi=100)
        plot_confusion_matrix(cnf_matrix, classes=self.num_class)
        CF_NonNormalized_filename = os.path.join(self.data.output_path, "Non_Normalized_" + str(self.epochs) + "_Epochs.eps")
        plt.savefig(CF_NonNormalized_filename, format='eps', dpi=500, bbox_inches="tight")

        # Plot normalized confusion matrix
        plt.figure(num=6, figsize=(5, 5), dpi=100)
        plot_confusion_matrix(cnf_matrix, classes=self.num_class, normalize=True)
        CF_Normalized_filename = os.path.join(self.data.output_path, "Normalized_" + str(self.epochs) + "_Epochs.eps")
        plt.savefig(CF_Normalized_filename, format='eps', dpi=500, bbox_inches="tight")
        cnf_matrix2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print(np.average(cnf_matrix2.diagonal()))

    def save(self):

        # serialize model to JSON
        model_json = self.net.to_json()
        with open(os.path.join(self.data.output_path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # evaluation on train
        # train_loss, train_acc, train_lr = self.net.evaluate(train_data, train_labels_one_hot_coded, batch_size=1)
        # print(train_loss, train_acc, train_lr)
        #
        # train_predicted_probs = self.net.predict(train_data, batch_size=1)
        # train_predicted_labels = train_predicted_probs.argmax(axis=-1)
        #
        # # Saving Train results
        # self.save_npy(train_predicted_probs, flag="Train", name="Probabilities")
        # self.save_npy(train_predicted_labels, flag="Train", name="ClassLabels")
        # self.save_csv(train_predicted_probs, flag="Train", name="Probabilities")
        # self.save_csv(train_predicted_labels, flag="Train", name="ClassLabels")
        #
        # # evaluation on Test
        # test_loss, test_acc, test_lr = self.net.evaluate(test_data, test_labels_one_hot_coded, batch_size=1)
        # print(test_loss, test_acc, test_lr)

        HyperParameter_Setting = self.collect_results()
        with open(os.path.join(self.data.output_path, "HyperParameters.txt"), "w") as text_file:
            text_file.write(HyperParameter_Setting)

        print(HyperParameter_Setting)
        shutil.copyfile(os.path.join(self.data.output_path, "models.py"),
                        os.path.join(self.data.output_path, "models.txt"))

    # saving labels or predicted probablities as a npy file
    def save_npy(self, data, flag="Train", name="Probabilities"):
        np.save(os.path.join(self.data.output_path, flag + "_" + name + "_" + str(self.epochs) + "_Epochs.npy"), data)

    # saving labels or predicted probablities as a csv file
    def save_csv(self, data, flag="Train", name="Probabilities"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data.output_path, flag + "_" + name + "_" + str(self.epochs) + "_Epochs.csv"))

    def save_layer_output(self, X, name="Train"):
        intermediate_layer_model = Model(inputs=self.net.input, outputs=self.net.get_layer(self.layer_name).output)
        intermediate_output = intermediate_layer_model.predict(X)
        filename = name + "_fc6_Layer_Features"
        np.save(os.path.join(self.data.output_path, filename), intermediate_output)

    def collect_results(self):
    # TODO: add calculation of union of interest in plots file.
        setting_info = "Saving folder Path =" + str(self.data.output_path)
        setting_info = setting_info + "\nData Path = " + str(self.data.train_img_path)
        # setting_info = setting_info + "\nSeed for Random Numbers = " + str(seednum)
        # setting_info = setting_info + "\nNumber of Folds = " + str(k)
        setting_info = setting_info + "\nNumber of Epochs In Training = " + str(self.epochs)
        setting_info = setting_info + "\nBatchsize = " + str(self.data.batch_size)
        setting_info = setting_info + "\nLearning Rate = " + str(self.lr)
        setting_info = setting_info + "\nFeatures Saved For Layer = " + str(self.layer_name)
        setting_info = setting_info + "\nCallbacks = " + self.callbacks
        setting_info = setting_info + "\nTrain accuracy = " + str(self.train_acc)
        setting_info = setting_info + "\nTrain loss = " + str(self.train_loss)
        setting_info = setting_info + "\nValidation accuracy = " + str(np.mean(self.vald_acc))
        setting_info = setting_info + "\nValidation loss = " + str(np.mean(self.vald_loss))
        setting_info = setting_info + "\nTest accuracy = " + str(self.test_acc)
        setting_info = setting_info + "\nTest loss = " + str(self.test_loss)
        setting_info = setting_info + "\nProcess Time in seconds = " + str(self.process_time)

    def tversky_loss(y_true, y_pred):
        alpha = 0.5
        beta = 0.5

        ones = K.ones(K.shape(y_true))
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = K.sum(p0 * g0, (0, 1, 2, 3))
        den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

        T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl - T

    def unet2d(self):
        # The original 2D UNET mdoel
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

    def unet3d(self):
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
