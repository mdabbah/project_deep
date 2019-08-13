import os

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.datasets import cifar100
from keras_contrib.applications import ResNet18
from distance_classifier import  DistanceClassifier
from keras.applications import Xception
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
datagen.fit(X_train)

batch_size = 100
num_epochs = 200

training_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
validation_generator = datagen.flow(X_valid, Y_valid, batch_size=batch_size, shuffle=True)

my_classifier = DistanceClassifier(input_size=(32, 32, 3), num_classes=100)

my_classifier.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

weights_folder = './new_arch_exp'

os.makedirs(weights_folder, exist_ok=True)
weights_file = f'{weights_folder}/' \
    '_{epoch: 03d}_{val_acc:.3f}_{val_loss:.3f}_{acc:.3f}_{loss:.3f}.h5'

# callbacks change if needed
csv_logger = CSVLogger(f'{weights_folder}.csv')
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')

my_callbacks = [csv_logger, model_checkpoint]  # lr_scheduler_callback , lr_reducer, early_stopper]

# start training
my_classifier.fit_generator(training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=len(training_generator),
                            callbacks=my_callbacks,
                            validation_data=validation_generator,
                            validation_steps=len(validation_generator),
                            workers=1,
                            use_multiprocessing=0)

