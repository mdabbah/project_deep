import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense
import os
from keras.losses import MSE


def distance_loss(encodeing_layer, batch_size=32, distance_margin = 25, distance_loss_coeff = 0.2,
                  training_type='distance_by_x_encoding'):
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
        batch_size = tf.shape(embeddings)[0]
        # filter out embeddings that had no associated regression target
        is_nan_mask = tf.is_nan(y_true)
        num_repeats = tf.shape(embeddings)[-1]//tf.shape(y_true)[-1]

        # calculate the embeddings distance from each other in the current batch
        squared_dists = tf.reduce_sum(tf.reshape(tf.squared_difference(K.expand_dims(embeddings, 0),
                                                                       tf.expand_dims(embeddings, 1)),
                                                 shape=(batch_size, batch_size, -1, num_repeats)), axis=-1)

        # filter embedding parts where label isn't well defined
        is_nan_mask = tf.logical_or(tf.expand_dims(is_nan_mask, 0), tf.expand_dims(is_nan_mask, 1))
        squared_dists = tf.where(is_nan_mask, tf.zeros_like(squared_dists), squared_dists)

        # the original classification loss
        total_loss = MSE_updated(y_true, y_pred)

        # the proposed distance loss
        distance_loss_eq = tf.reduce_sum(squared_dists)
        # print them
        print_op2 = tf.print("\nloss equals: \n", distance_loss_eq)
        print_embedd_size = tf.print('embedding_size \n', tf.shape(embeddings))

        total_loss += distance_loss_eq  * distance_loss_coeff
        with tf.control_dependencies([print_op2, print_embedd_size]):
            return total_loss

    def loss_by_x_embedding(y_true, y_pred):

        # batch_size = int(y_true.shape[0])
        embeddings = encodeing_layer.output

        # calculate the embeddings distance from each other in the current batch
        squared_dists = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0),
                                                            tf.expand_dims(embeddings, 1)), axis=-1)

        # the original regression loss
        total_loss = MSE_updated(y_true, y_pred)

        # the proposed distance loss
        distance_loss_eq = tf.reduce_mean(squared_dists)
        # print them
        print_op2 = tf.print("\nloss equals: \n", distance_loss_eq)
        print_embedd_size = tf.print('embedding_size \n', tf.shape(embeddings))

        total_loss += distance_loss_eq * distance_loss_coeff
        with tf.control_dependencies([print_op2, print_embedd_size]):
            return total_loss

    if training_type == 'distance_by_x_encoding':
        return loss_by_x_embedding

    return loss


def embeddings_avg_distance_metric(embedding_layer, metric: str):
    """
    returns a method that calculates the average/std distance between the outputs
    of the given layer
    :param metric: one of 'mean' or 'std'
    :param embedding_layer: the desired layer of a model
    :return: a method to be passed to model metrics
    """

    def avg_metric(y_true, y_pred):
        # we will totally ignore the y_true and y_predas we are only interested in the outputs of the
        # embedding layer
        embeddings = embedding_layer.output

        # calculate the embeddings distance from each other in the current batch
        squared_dists = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0),
                                                            tf.expand_dims(embeddings, 1)), axis=-1)
        return tf.reduce_mean(squared_dists)

    def std_metric(y_true, y_pred):
        # we will totally ignore the y_true and y_predas we are only interested in the outputs of the
        # embedding layer
        embeddings = embedding_layer.output

        # calculate the embeddings distance from each other in the current batch
        squared_dists = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0),
                                                            tf.expand_dims(embeddings, 1)), axis=-1)
        return tf.math.reduce_std(squared_dists)

    if metric == 'mean':
        return avg_metric
    elif metric == 'std':
        return std_metric
    elif metric == 'both':
        return [avg_metric, std_metric]


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


def MSE_updated(y_true, y_pred, return_vec: bool = False):
    """
    same as MSE but ignores cells where y_true is nan
    :param return_vec: whether to return mean loss on samples or vector of losses on each sample
    :param y_true: true regression numbers
    :param y_pred: net work output
    :return:
    """

    mask = 1-tf.cast(tf.is_nan(y_true), y_true.dtype)
    num_non_nans = tf.reduce_sum(mask, axis=-1)
    mask = ~tf.is_nan(y_true)
    y_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)  # to avoid nan gradients when using tf.where

    squared_loss = tf.square(y_true-y_pred)
    squared_loss_filtered = tf.where(mask, squared_loss, tf.zeros_like(mask, dtype=y_true.dtype))

    total_loss = tf.math.divide(tf.reduce_sum(squared_loss_filtered, axis=-1), num_non_nans)

    # # debugging mse updated
    # print_non_nans = tf.print('\nmin num_non_nans:\n', num_non_nans)
    #
    # orig_mse_print_node = tf.print('orig mse:\n', MSE(y_true, y_pred))
    # my_mse_print_node = tf.print('my mse:\n', total_loss)
    # print_node = tf.print('IS EQUAL == \n', 1 - tf.math.reduce_sum(tf.math.abs(MSE(y_true, y_pred)- total_loss)))
    # with tf.control_dependencies([print_node, print_non_nans, my_mse_print_node, orig_mse_print_node]):
    #     return tf.math.reduce_mean(total_loss)+ 0.
    if return_vec:
        return total_loss
    return tf.math.reduce_mean(total_loss)


loss_functions_cache = {'MSE': MSE, 'MSE_updated': MSE_updated, 'l1_smooth_loss_updated': l1_smooth_loss_updated,
                        'l1_smooth_loss': tf.losses.huber_loss}

if __name__ == '__main__':

    # wheere to save weights , dataset & training details change if needed
    data_set = 'facial_key_points'
    training_type = 'distance_by_x_encoding'  # options 'l1_smoothed', 'distance_classifier'
    arch = 'facial_key_points_arc'
    weights_folder = f'./results/regression/{training_type}s_{arch}_{data_set}'
    os.makedirs(weights_folder, exist_ok=True)
    weights_file = f'{weights_folder}/{training_type}_{arch}' \
                   '_{epoch: 03d}_{val_loss:.3f}_{loss:.3f}_{MSE_updated:.5f}_{val_MSE_updated: .5f}.h5'

    # callbacks change if needed
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)

    # lr_scheduler_callback = LearningRateScheduler(lr_scheduler_maker(data_set))
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger(f'{training_type}-{data_set}-{arch}.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_MSE_updated', save_best_only=True,
                                       save_weights_only=True, mode='min')
    my_callbacks = [csv_logger, model_checkpoint]  # lr_scheduler_callback , lr_reducer, early_stopper]

    # loading data
    if data_set == 'facial_key_points':
        from data_generators import facial_keypoints_data_generator as data_genetator

        print("training on facial keypoints")
    elif data_set == 'fingers':
        raise ValueError("not supported yet")

    # training constants, change if needed
    batch_size = 32
    num_epochs = 600
    # distance_margin = 25
    distance_loss_coeff = 0.0007
    shuffle = True
    input_size = (96, 96, 3)
    my_regressor = None
    use_nans = True
    flip_prob = 0.3
    my_training_generator = None
    my_validation_generator = None

    # choosing arch and optimizer
    if data_set.startswith('facial'):
        if arch.startswith('ELU_arc'):
            from models.distance_classifier import DistanceClassifier
            base_model = DistanceClassifier(input_size, num_classes=None, include_top=False)
            x = base_model.output
            x = Dense(30, activation='linear')(x)
            my_regressor = Model(base_model.input, x, name=f'{data_set} regression model')
        else:
            from models.facial_keypoints_arc import FacialKeypointsArc
            my_regressor = FacialKeypointsArc(input_size, 30, 480)

        my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=shuffle,
                                                           use_nans=use_nans, horizontal_flip_prob=flip_prob)
        my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=shuffle,
                                                             use_nans=use_nans, horizontal_flip_prob=flip_prob)
    elif data_set.startswith('concrete'):
        from data_generators import concrete_dataset_generator as data_genetator
        from models.concrete_strength_arc import simple_FCN
        my_regressor = simple_FCN(8, 1)

        num_epochs = 800
        batch_size = 32
        my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=True)
        my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=True)
    optimizer = 'adadelta'

    num_training_xsamples_per_epoch = len(my_training_generator)
    num_validation_xsamples_per_epoch = len(my_validation_generator)

    encoder = my_regressor.get_layer('embedding')
    if training_type.startswith('distance'):
        loss_function = distance_loss(encoder, batch_size,
                                      distance_loss_coeff=distance_loss_coeff,  training_type=training_type)
    else:
        loss_function = loss_functions_cache[training_type]

    my_regressor.compile(optimizer=optimizer, loss=loss_function, metrics=[MSE_updated, *embeddings_avg_distance_metric(encoder, 'both')])

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

