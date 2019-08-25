from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Lambda, Flatten, ELU, LeakyReLU, BatchNormalization
import keras.models as models
from keras import regularizers
import keras.backend as K


def simple_FCN(input_size, num_output_neurons, include_top=True, mc_dropout_rate=0.):


    input = Input(shape=(input_size,))

    x = Dense(units=64, activation='relu', name='hidden_1')(input)
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(units=16, activation='relu', name='embedding')(x)
    x = Lambda(lambda l_in: K.dropout(l_in, level=mc_dropout_rate), name='drop_out_to_turn_on')(x)

    if include_top:
        x = Dense(num_output_neurons, activation='linear')(x)

    # Create model.
    model = models.Model(input, x, name='distance_predictor')

    return model
