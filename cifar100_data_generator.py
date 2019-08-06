import os
import keras
from keras.datasets import cifar100
from keras.utils import np_utils
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from skimage.transform import resize

# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
nb_classes = 100


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


def rearrange_samples(X_train, Y_train, nb_classes):
    """rearranges samples so it guaranties that 20% of pairs in each
    batch is of the same class"""

    # rearrange the dataset so it has the samples in the following order
    # AAAA...BBBB...CCCC... where A,B,C ... are our classes
    X_train = X_train[np.argsort(y_train, axis=0), :, :, :]
    Y_train = Y_train[np.argsort(y_train, axis=0), :]

    x_train_copy = np.copy(np.squeeze(X_train))
    y_train_copy = np.copy(np.squeeze(Y_train))
    class_size = 500
    num_samples_per_class_in_bach = 20
    x_train_copy = np.reshape(x_train_copy, [nb_classes, class_size, 32, 32, 3])
    y_train_copy = np.reshape(y_train_copy, [nb_classes, class_size, nb_classes])

    randomize_class_order = np.random.permutation(nb_classes)
    x_train_copy = x_train_copy[randomize_class_order, :, :, :, :]
    y_train_copy = y_train_copy[randomize_class_order, :, :]

    randomize_samples_order = np.random.permutation(class_size)
    x_train_copy = x_train_copy[:, randomize_samples_order, :, :, :]
    y_train_copy = y_train_copy[:, randomize_samples_order, :]

    x_train_copy = np.reshape(x_train_copy, [-1, num_samples_per_class_in_bach, 32, 32, 3])
    y_train_copy = np.reshape(y_train_copy, [-1, num_samples_per_class_in_bach, nb_classes])
    randomize_20 = np.random.permutation(x_train_copy.shape[0])
    x_train_copy = x_train_copy[randomize_20, :, :, :, :]
    y_train_copy = y_train_copy[randomize_20, :, :]
    X_train = np.reshape(x_train_copy, [-1, 32, 32, 3])
    Y_train = np.reshape(y_train_copy, [-1, nb_classes])

    return X_train, Y_train


np.random.seed(0)
X_train, Y_train = rearrange_samples(X_train, Y_train, nb_classes)


# split data for validation
num_training_samples = int(X_train.shape[0]*0.9)
X_valid = X_train[num_training_samples:, :, :, :]
X_train = X_train[:num_training_samples, :, :, :]

Y_valid = Y_train[num_training_samples:, :]
Y_train = Y_train[:num_training_samples, :]


# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    samplewise_center=False,  # set each sample mean to 0
    featurewise_center=True,
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


def identity_preprocessing(x):
    return x


class MYGenerator(keras.utils.Sequence):

    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False,
                 preprocessing_fun=None, input_size=(32, 32, 3)):

        global X_train, X_test, X_valid
        if data_type == 'train':
            self.imgs = X_train
            self.labels = Y_train
        elif data_type == 'valid':
            self.imgs = X_valid
            self.labels = Y_valid
        else:
            self.imgs = X_test
            self.labels = Y_test

        if preprocessing_fun is None:
            # subtract mean and normalize -- global preprocessing
            mean_image = np.mean(X_train, axis=0)
            X_train -= mean_image
            X_valid -= mean_image
            X_test -= mean_image
            X_train /= 128.
            X_valid /= 128.
            X_test /= 128.
            preprocessing_fun = identity_preprocessing

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(X_train)

        self.pre_processing_fun = preprocessing_fun
        self.img_size = input_size

        if shuffle:
            size = self.imgs.shape[0]
            permute = np.random.permutation(size)
            self.imgs = self.imgs[permute, :, :, :]
            self.labels = self.labels[permute, :]

        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.imgs[idx * self.batch_size: (idx + 1) * self.batch_size]
        if self.pre_processing_fun == identity_preprocessing:
            batch_x = next(datagen.flow(batch_x, None, batch_size=self.batch_size, shuffle=False))

        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        return resize(self.pre_processing_fun(batch_x), (batch_x.shape[0], *self.img_size)), batch_y
