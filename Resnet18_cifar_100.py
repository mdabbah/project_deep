import os

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.datasets import cifar100
from keras.optimizers import SGD
from resnet_keras_contrib import ResNet18
from keras.utils import np_utils
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy

# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
nb_classes = 100


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


np.random.seed(0)

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
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    zoom_range=0.05,
)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

batch_size = 100
num_epochs = 200

training_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
validation_generator = datagen.flow(X_valid, Y_valid, batch_size=batch_size, shuffle=True)

my_classifier = ResNet18((32, 32, 3), 100)
optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
my_classifier.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

weights_folder = './resnet18_exp_sgd_no_zca_30'

os.makedirs(weights_folder, exist_ok=True)
weights_file = f'{weights_folder}/' \
    '_{epoch: 03d}_{val_acc:.3f}_{val_loss:.3f}_{acc:.3f}_{loss:.3f}.h5'

# callbacks change if needed
csv_logger = CSVLogger(f'{weights_folder}.csv')
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')


def lr_scheduler(epoch, current_lr):
    """
    the function used to decrease the learning rate as suggested by
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param epoch: the current epoch
    :param current_lr: the current learning rate
    :return: the new learning rate
    """
    return current_lr / 5 ** (epoch // 30)


my_callbacks = [csv_logger, model_checkpoint, LearningRateScheduler(lr_scheduler)]  # lr_scheduler_callback , lr_reducer, early_stopper]

# start training
my_classifier.fit_generator(training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=len(training_generator),
                            callbacks=my_callbacks,
                            validation_data=validation_generator,
                            validation_steps=len(validation_generator),
                            workers=1,
                            use_multiprocessing=0)

