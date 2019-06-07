import distance_classifier
import cifar100_data_generator
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
import sys


def distance_loss(encodeing_layer):

    def loss(y_true, y_pred):
        # print_op = tf.print("weights: ", [encodeing_layer.output], output_stream=sys.stdout)
        emebddings = K.squeeze(encodeing_layer.output, 1)
        emebddings = K.squeeze(emebddings, 1)
        norms = tf.norm(K.repeat(emebddings, batch_size) - tf.transpose(K.repeat(emebddings, batch_size), [1, 0, 2]), axis=2)
        total_loss = K.categorical_crossentropy(y_true, y_pred)

        print_op = tf.print("y_true: ", K.int_shape(y_true))
        print_op_pred = tf.print("y pred: ", K.int_shape(y_pred))
        print_op2 = tf.print("y_true: ", y_true)
        # y_true = K.squeeze(K.squeeze(y_true, 1), 1)
        # y_true = tf.math.argmax(y_true, -1)
        #
        # y_true = tf.expand_dims(y_true, 1)
        if K.int_shape(y_true)[1] == 1 or K.int_shape(y_true)[1] == None:
            y_true = tf.squeeze(y_true, 1)
            y_true = tf.squeeze(y_true, 1)
        y_eq = tf.matmul(y_true, tf.transpose(y_true)) > 0
        # y_eq = tf.equal(y_true, K.transpose(y_true))

        distance_loss_eq = tf.reduce_sum(tf.boolean_mask(norms, y_eq))
        distance_loss_diff = tf.reduce_sum(tf.maximum(0., distance_margin - tf.boolean_mask(norms, tf.logical_not(y_eq))))

        # same_class_cntr = 0
        # for i in range(batch_size):
        #     i_same_class = y_true == y_true[i]
        #     loss += norms[i, i_same_class]
        #     same_class_cntr = np.sum(i_same_class)
        #     loss -= tf.maximum(0, distance_margin - norms[i, ~i_same_class])

        with tf.control_dependencies([print_op, print_op2, print_op_pred]):
            return total_loss + distance_loss_eq

    return loss


if __name__ == '__main__':

    weights_file = 'distance_classifier_{epoch}_{vall_acc}_{vall_loss}_{train_acc}_{train_loss}.h5'
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger('distance_classifier-CIFAR-10.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                       save_weights_only=True, mode='auto')

    my_callbacks = [csv_logger, model_checkpoint, lr_reducer, early_stopper]
    batch_size = 100
    distance_margin = 25
    distance_loss_coeff = 0.2

    my_generator = cifar100_data_generator.datagen.flow(cifar100_data_generator.X_train,
                                                        cifar100_data_generator.Y_train,
                                                        batch_size=batch_size)

    c100_classifier = distance_classifier.DistanceClassifier((32, 32, 3), num_classes=100)

    validation_data = cifar100_data_generator.X_valid, cifar100_data_generator.Y_valid

    num_training_xsamples_per_epoch = cifar100_data_generator.X_train.shape[0] // batch_size
    num_validation_xsamples_per_epoch = cifar100_data_generator.X_valid.shape[0] // batch_size

    encoder = c100_classifier.get_layer('embedding')

    c100_classifier.compile(optimizer='Nadam', loss=distance_loss(encoder))
    c100_classifier.fit(my_generator,
                                  steps_per_epoch=num_training_xsamples_per_epoch,
                                  callbacks=my_callbacks,
                                  validation_data=validation_data,
                                  validation_steps=num_validation_xsamples_per_epoch,
                                  workers=1,
                                  use_multiprocessing=False)

