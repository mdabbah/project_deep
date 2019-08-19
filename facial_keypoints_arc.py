from keras.layers import Input, Dense, Conv2D, MaxPool2D, LeakyReLU, Dropout, Lambda, Flatten, ELU
import keras.models as models
from keras import regularizers


def FacialKeypointsArc(input_size, num_targets, num_last_hidden_units=510):

    img_input = Input(shape=input_size)

    # first conv block
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x = ELU(alpha=0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x)

    # second conv block
    x = Conv2D(filters=64, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    # 3rd conv block
    x = Conv2D(filters=128, kernel_size=(2, 2), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = ELU(alpha=0.2)(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.3)(x)

    # 4th block
    x = Dense(num_last_hidden_units)(x)
    x = ELU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # 5th block
    x = Dense(name='embedding', units=num_last_hidden_units)(x)
    x = ELU(alpha=0.2)(x)
    x = Dense(num_targets, activation='linear')(x)
    # Create model.
    model = models.Model(img_input, x, name='distance_predictor')

    return model
