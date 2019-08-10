import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, SGD
import keras.applications.inception_resnet_v2 as inception_resnet_v2
from keras_contrib.applications.resnet import ResNet18
import os


def distance_loss(encodeing_layer, batch_size=100, distance_margin = 25, distance_loss_coeff = 0.2):
    """
    the loss function that depends on the encoding of the second to last layer
    as suggested by
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param encodeing_layer: which layer we want to use as encoder
    :return: loss function
    """
    print("generating distance loss function ...")
    def loss(y_true, y_pred):

        # batch_size = int(y_true.shape[0])
        embeddings = encodeing_layer.output
        eps = 0.0015
        # calculate the embeddings distance from each other in the current batch
        # norms = tf.norm(K.expand_dims(embeddings, 0)``1vcxc>\ - tf.expand_dims(embeddings, 1), axis=2)
        norms = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0), tf.expand_dims(embeddings, 1)), axis=2)
        norms = tf.sqrt(norms + eps)

        # no sqrt implementation:
        # norms = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0), tf.expand_dims(embeddings, 1)), axis=2)

        # the original classification loss
        total_loss = K.categorical_crossentropy(y_true, y_pred)
        total_loss = tf.reduce_mean(total_loss)
        print_op_pred = tf.print("classification loss: ", total_loss)  # print it

        # boolean matrix s.t. in entry (i,j) is 1 iff label_i == label_j
        y_eq = tf.matmul(y_true, tf.transpose(y_true))
        num_pairs = tf.cast((tf.shape(y_eq)[0])**2, tf.float32)
        print_op = tf.print("eq. pairs percentage: ", tf.reduce_sum(y_eq)/num_pairs)  # print how manny pairs are equal
        print_batch_sz = tf.print("num pairs", num_pairs)

        # the proposed distance loss
        distance_loss_eq = tf.reduce_sum(tf.boolean_mask(norms, y_eq - tf.eye(tf.shape(y_eq)[0], dtype=tf.float32)))/num_pairs  # loss for pairs with the same label
        distance_loss_diff = tf.reduce_sum(tf.maximum(0., distance_margin - tf.boolean_mask(norms, 1-y_eq)))/num_pairs  # loss for pairs with different label
        # print them
        print_op2 = tf.print("loss equals: ", distance_loss_eq)
        print_op3 = tf.print("loss diff: ", distance_loss_diff)

        total_loss += (distance_loss_eq + distance_loss_diff) * distance_loss_coeff
        with tf.control_dependencies([print_op, print_op2, print_op3, print_batch_sz]):
            return total_loss

    return loss


def l1_smooth_loss_updated(y_true, y_pred):
    """
    same as l1_smooth_loss but ignores cells where y_true is nan
    :param y_true: true regression numbers
    :param y_pred: net work output
    :return:
    """

    mask = 1-tf.cast(tf.is_nan(y_true), tf.float64)
    num_non_nans = tf.reduce_sum(mask, axis=-1)
    mask = ~tf.is_nan(y_true)

    abs_loss = tf.abs(y_true-y_pred)-0.5
    squared_loss = 0.5*tf.square(y_true-y_pred)
    l1_smoothed = tf.where(mask, tf.where(abs_loss <= 1, squared_loss, abs_loss), tf.zeros_like(mask, dtype=tf.float64))

    total_loss = tf.math.divide(tf.reduce_sum(l1_smoothed, axis=-1), num_non_nans)
    return tf.math.reduce_mean(total_loss)


def MSE_updated(y_true, y_pred):
    """
    same as MSE but ignores cells where y_true is nan
    :param y_true: true regression numbers
    :param y_pred: net work output
    :return:
    """

    mask = 1-tf.cast(tf.is_nan(y_true), tf.float64)
    num_non_nans = tf.reduce_sum(mask, axis=-1)
    mask = ~tf.is_nan(y_true)

    squared_loss = tf.square(y_true-y_pred)
    squared_loss_filtered = tf.where(mask, squared_loss, tf.zeros_like(mask, dtype=tf.float64))

    total_loss = tf.math.divide(tf.reduce_sum(squared_loss_filtered, axis=-1), num_non_nans)
    return tf.math.reduce_mean(total_loss)


if __name__ == '__main__':

    # wheere to save weights , dataset & training details change if needed
    data_set = 'facial_key_points'
    training_type = 'l1_smoothed'  # options 'l1_smoothed', 'distance_classifier'
    arch = 'ELU_arch'
    weights_folder = f'./results/regression/{training_type}s_{arch}_{data_set}'
    os.makedirs(weights_folder, exist_ok=True)
    weights_file = f'{weights_folder}/{training_type}_{arch}' \
                   '_{epoch: 03d}_{val_loss:.3f}_{loss:.3f}.h5'

    # callbacks change if needed
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)

    # lr_scheduler_callback = LearningRateScheduler(lr_scheduler_maker(data_set))
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger(f'{training_type}-{data_set}-{arch}.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True,
                                       save_weights_only=True, mode='auto')
    my_callbacks = [csv_logger, model_checkpoint] # lr_scheduler_callback , lr_reducer, early_stopper]

    # loading data
    if data_set == 'facial_key_points':
        import facial_keypoints_data_generator as data_genetator  # choose data set
        print("training on facial keypoints")
    elif data_set == 'fingers':
        raise ValueError("not supported yet")

    # training constants, change if needed
    batch_size = 64
    num_epochs = 3000
    # distance_margin = 25
    # distance_loss_coeff = 0.2
    shuffle = True
    input_size = (96, 96, 3)
    my_regressor = None
    use_nans = False

    # choosing arch and optimizer
    if data_set.startswith('facial'):
        from distance_classifier import DistanceClassifier
        base_model = DistanceClassifier(input_size, num_classes=None, include_top=False)
        x = base_model.output
        x = Dense(30, activation='linear')(x)
        my_regressor = Model(base_model.input, x, name=f'{data_set} regression model')
    optimizer = 'adadelta'


    # data generators
    my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=shuffle,
                                                       use_nans=use_nans)
    my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=shuffle,
                                                         use_nans=use_nans)

    num_training_xsamples_per_epoch = len(my_training_generator)
    num_validation_xsamples_per_epoch = len(my_validation_generator)

    encoder = my_regressor.get_layer('embedding')
    loss_function = \
        distance_loss(encoder, batch_size) if training_type.startswith('distance') else l1_smooth_loss_updated

    my_regressor.compile(optimizer=optimizer, loss=loss_function, metrics=['MSE'])

    # start training
    my_regressor.fit_generator(my_training_generator,
                               epochs=num_epochs,
                               steps_per_epoch=num_training_xsamples_per_epoch,
                               callbacks=my_callbacks,
                               validation_data=my_validation_generator,
                               validation_steps=num_validation_xsamples_per_epoch,
                               workers=1,
                               use_multiprocessing=0)

    test_generator = data_genetator.MYGenerator(data_type='test', batch_size=batch_size, shuffle=True,
                                                use_nans=True, horizontal_flip_prob=0)

    # check acc
    loss, acc = my_regressor.evaluate_generator(test_generator,
                                                steps=len(test_generator))

    print(f'test acc {acc}, test loss {loss}')

