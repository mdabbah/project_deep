from keras.layers import Input, Dense, Conv2D, MaxPool2D, LeakyReLU, Dropout, Lambda, Flatten, ELU, concatenate
import keras.models as models
from keras import regularizers
import keras.backend as K


def FacialKeypointsArc(input_size, num_targets, num_last_hidden_units=510):

    version = 'v1'
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
    x = Flatten()(x)
    x = Dense(num_last_hidden_units)(x)
    x = ELU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    # 5th block
    if version == 'v1':
        x = Dense(name='embedding', units=num_last_hidden_units)(x)
        x = ELU(alpha=0.2)(x)
        x = Lambda(lambda l_in: K.dropout(l_in, level=0.), name='drop_out_to_turn_on')(x)
        x = Dense(num_targets, activation='linear')(x)
        # Create model.
        model = models.Model(img_input, x, name='distance_predictor')
    else:
        outputs = []
        before_branching = x
        for yi in range(num_targets):
            x = Dense(name='embedding_'+str(yi), units=num_last_hidden_units//num_targets)(x)
            x = ELU(alpha=0.2)(x)
            x = Lambda(lambda l_in: K.dropout(l_in, level=0.), name='drop_out_to_turn_on'+str(yi))(x)
            x = Dense(1, activation='linear')(x)
            outputs.append(x)
            x = before_branching

        output = concatenate(outputs)
        # Create model.
        model = models.Model(img_input, output, name='distance_predictor')

    return model
