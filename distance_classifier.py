from keras.layers import Input, Dense, Conv2D, MaxPool2D, ELU, Dropout, Lambda, Flatten
import keras.models as models
from keras import regularizers


def DistanceClassifier(input_size, num_classes, include_top=True):

    img_input = Input(shape=input_size)

    # first conv block
    x = Conv2D(filters=192, kernel_size=(5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x = ELU(alpha=0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # second conv block
    x = Conv2D(filters=192, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)

    # 3rd conv block
    x = Conv2D(filters=240, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.1)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # 4th conv block
    x = Conv2D(filters=240, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)

    # 5th conv block
    x = Conv2D(filters=260, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # 6th conv block
    x = Conv2D(filters=260, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)

    # 7th conv block
    x = Conv2D(filters=280, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # 8th conv block
    x = Conv2D(filters=280, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)

    # 9th conv block
    x = Conv2D(filters=300, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # 10th conv block
    x = Conv2D(filters=300, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Flatten(name='embedding')(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = Dense(num_classes, activation='softmax')(x)

    # Create model.
    model = models.Model(img_input, x, name='distance_predictor')

    return model
