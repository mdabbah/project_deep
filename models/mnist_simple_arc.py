
# Importing the required Keras modules containing model and layers
from keras import Model, Input, Sequential
from keras.engine import InputLayer
import keras.backend as K
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Lambda, BatchNormalization


# Creating a Sequential Model and adding the layers


def mnist_simple_arc(input_shape=(28, 28, 1), num_targets=1, final_activation='linear', mc_dropout_rate=0.):

    input = Input(shape=input_shape)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Lambda(lambda l_in: K.dropout(l_in, level=mc_dropout_rate), name='embedding'))
    model.add(Dense(num_targets, activation=final_activation))

    return model
