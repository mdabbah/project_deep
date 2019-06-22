from skimage.io import imread
from skimage.transform import resize
import keras
import stl10_lib
from keras.utils import np_utils
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from itertools import count
from sklearn.metrics import roc_auc_score

# load data
from approx import batch_size

(X_train, y_train), (X_test, y_test) = stl10_lib.load_data()

nb_classes = 10


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


def get_file_paths(data, labels, type):

    file_paths = []
    for id, (sample, label) in enumerate(zip(data, labels)):
        file_paths.append(f'./data/stl10/{type}/{label+1}/{id}.png')

    return np.array(file_paths)


train_filepaths = get_file_paths(X_train, y_train, 'train')
test_filepaths = get_file_paths(X_test, y_test, 'test')


def rearrange_samples(X_train, Y_train, file_paths, nb_classes):
    """rearranges samples so it guaranties that 20% of pairs in each
    batch is of the same class"""

    # rearrange the dataset so it has the samples in the following order
    # AAAA...BBBB...CCCC... where A,B,C ... are our classes
    arg_sort_perm = np.argsort(y_train, axis=0)
    X_train = X_train[arg_sort_perm, :, :, :]
    Y_train = Y_train[arg_sort_perm, :]
    file_paths = file_paths[arg_sort_perm]

    x_train_copy = np.copy(np.squeeze(X_train))
    y_train_copy = np.copy(np.squeeze(Y_train))
    file_paths_copy = np.copy(file_paths)

    class_size = 500
    num_samples_per_class_in_bach = 20
    x_train_copy = np.reshape(x_train_copy, [nb_classes, class_size, 96, 96, 3])
    y_train_copy = np.reshape(y_train_copy, [nb_classes, class_size, nb_classes])
    file_paths_copy = np.reshape(file_paths_copy, [nb_classes, class_size, 1])

    randomize_class_order = np.random.permutation(nb_classes)
    x_train_copy = x_train_copy[randomize_class_order, :, :, :, :]
    y_train_copy = y_train_copy[randomize_class_order, :, :]
    file_paths_copy = file_paths_copy[randomize_class_order, :, :]

    randomize_samples_order = np.random.permutation(class_size)
    x_train_copy = x_train_copy[:, randomize_samples_order, :, :, :]
    y_train_copy = y_train_copy[:, randomize_samples_order, :]
    file_paths_copy = file_paths_copy[:, randomize_samples_order, :]

    x_train_copy = np.reshape(x_train_copy, [-1, num_samples_per_class_in_bach, 96, 96, 3])
    y_train_copy = np.reshape(y_train_copy, [-1, num_samples_per_class_in_bach, nb_classes])
    file_paths_copy = np.reshape(file_paths_copy, [-1, num_samples_per_class_in_bach, 1])

    randomize_20 = np.random.permutation(x_train_copy.shape[0])
    x_train_copy = x_train_copy[randomize_20, :, :, :, :]
    y_train_copy = y_train_copy[randomize_20, :, :]
    file_paths_copy = file_paths_copy[randomize_20, :,:]

    X_train = np.reshape(x_train_copy, [-1, 96, 96, 3])
    Y_train = np.reshape(y_train_copy, [-1, nb_classes])
    file_paths = np.reshape(file_paths_copy, [-1, 1])
    return X_train, Y_train, file_paths


np.random.seed(0)
X_train, Y_train, train_filepaths = rearrange_samples(X_train, Y_train, train_filepaths, nb_classes)

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

# split data for validation
num_training_samples = int(X_train.shape[0]*0.9)
X_valid = X_train[num_training_samples:, :, :, :]
X_train = X_train[:num_training_samples, :, :, :]


Y_valid = Y_train[num_training_samples:, :]
Y_train = Y_train[:num_training_samples, :]

valid_filepaths = train_filepaths[num_training_samples:]
train_filepaths = train_filepaths[:num_training_samples]

del X_train, X_test, X_valid

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

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
# datagen.fit(X_train)


def preprocess_fun(img_batch):
    img_batch = img_batch.astype('float32')
    img_batch -= mean_image
    img_batch /= 128.

    return img_batch


class MYGenerator(keras.utils.Sequence):

    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False):

        if data_type == 'train':
            self.labels = Y_train
            self.file_paths = train_filepaths
        elif data_type == 'valid':
            self.labels = Y_valid
            self.file_paths = valid_filepaths
        else:
            self.labels = Y_test
            self.file_paths = test_filepaths

        if shuffle:
            size = self.labels.shape[0]
            permute = np.random.permutation(size)
            self.labels = self.labels[permute, :]
            self.file_paths = self.file_paths[permute]

        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        #
        # batch_x = next(datagen.flow(self.imgs[idx * self.batch_size: (idx + 1) * self.batch_size]
        #                             , None, batch_size=self.batch_size, shuffle=False))\

        batch_x = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size, :]
        batch_x = np.array([
            preprocess_fun(imread(file_name[0]))
            for file_name in batch_x])

        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        return batch_x, batch_y
