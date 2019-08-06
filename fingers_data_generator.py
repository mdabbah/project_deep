import os
import keras
from keras.utils import np_utils
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize



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

    DATAPATH=r'.\data\regression\fingers'
    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False,
                 preprocessing_fun=None, input_size=(32, 32, 3)):

        self.data_path = f'{self.DATAPATH}/{data_type}'
        self.img_names = np.ndarray(os.listdir(self.data_path))
        self.pre_processing_fun = preprocessing_fun
        self.img_size = input_size

        if shuffle:
            size = self.img_names.shape[0]
            permute = np.random.permutation(size)
            self.img_names = self.img_names[permute]

        self.labels = [float(img.split('.png')[0][-2:-1]) for img in self.img_names]
        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = [imread(os.path.join(self.data_path, im_name))
                   for im_name in self.img_names[idx * self.batch_size: (idx + 1) * self.batch_size]]

        if self.pre_processing_fun == identity_preprocessing:
            batch_x = next(datagen.flow(batch_x, None, batch_size=self.batch_size, shuffle=False))

        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        return resize(self.pre_processing_fun(batch_x), (batch_x.shape[0], *self.img_size)), batch_y
