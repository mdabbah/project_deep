import keras
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from itertools import count
from sklearn.model_selection import train_test_split

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
input_shape = (28, 28, 1)
nb_classes = 10
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_test = Y_test.reshape(Y_test.shape[0] , 1)


def normalize(X_train, X_test):
    '''
    This function normalize inputs for zero mean and unit variance
    Args:
        X_train: np array of train samples, axis 0 is samples.
        X_test: np array of test/validation samples, axis 0 is samples.
    Returns:
        A tuple (X_train, X_test), Normalized version of the data.
    '''
    X_train /= 255.
    X_test /= 255.
    return X_train, X_test


X_train, X_test = normalize(X_train, X_test)

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    samplewise_center=False,  # set each sample mean to 0
    featurewise_center=False,
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

# split data for validation
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=0.1,
                                                      random_state=1)


class MYGenerator(keras.utils.Sequence):

    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False, input_size=(28, 28, 1),
                 augment=False):

        global X_train, X_test, X_valid
        self.augment = augment
        self.dataset_name = 'mnist'
        if data_type == 'train':
            self.imgs = X_train
            self.gt = Y_train
            self.augment = True
        elif data_type == 'valid':
            self.imgs = X_valid
            self.gt = Y_valid
        else:
            self.imgs = X_test
            self.gt = Y_test

        self.img_size = input_size

        if shuffle:
            size = self.imgs.shape[0]
            permute = np.random.permutation(size)
            self.imgs = self.imgs[permute, :, :, :]
            self.gt = self.gt[permute, :]

        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.gt) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.imgs[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.augment:
            batch_x = next(datagen.flow(batch_x, None, batch_size=self.batch_size, shuffle=False))
        batch_y = self.gt[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        # return resize(self.pre_processing_fun(batch_x), (batch_x.shape[0], *self.img_size)), batch_y
        return batch_x, batch_y