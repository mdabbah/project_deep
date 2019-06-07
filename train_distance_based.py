import distance_classifier
import cifar100_data_generator
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import sys


def distance_loss(encodeing_layer):

    def loss(y_true, y_pred):
        emebddings = K.squeeze(encodeing_layer.output, 1)
        emebddings = K.squeeze(emebddings, 1)
        norms = tf.norm(K.repeat(emebddings, batch_size) - tf.transpose(K.repeat(emebddings, batch_size), [1, 0, 2]), axis=2)
        total_loss = K.categorical_crossentropy(y_true, y_pred)

        print_op_pred = tf.print("total loss: ", total_loss)

        y_eq = tf.matmul(y_true, tf.transpose(y_true)) > 0
        # print_op = tf.print("eq: ", tf.reduce_sum(tf.matmul(y_true, tf.transpose(y_true)))/10000)

        distance_loss_eq = tf.reduce_sum(tf.boolean_mask(norms, y_eq))
        print_op2 = tf.print("loss equals: ", distance_loss_eq)
        distance_loss_diff = tf.reduce_sum(tf.maximum(0., distance_margin - tf.boolean_mask(norms, tf.logical_not(y_eq))))
        print_op3 = tf.print("loss diff: ", distance_loss_diff)

        with tf.control_dependencies([print_op2, print_op_pred, print_op3]):
            return total_loss + (distance_loss_eq + distance_loss_diff)*distance_loss_coeff

    return loss


if __name__ == '__main__':

    weights_file = 'distance_classifier_{epoch: 03d}_{vall_acc:.3f}_{vall_loss:.3f}_{train_acc:.3f}_{train_loss:.3f}.h5'
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

    # y_train = np.expand_dims(cifar100_data_generator.Y_train, axis=1)
    # y_train = np.expand_dims(y_train, axis=1)
    my_generator = cifar100_data_generator.datagen.flow(cifar100_data_generator.X_train,
                                                        cifar100_data_generator.Y_train,
                                                        batch_size=batch_size)

    c100_classifier = distance_classifier.DistanceClassifier((32, 32, 3), num_classes=100)

    validation_data = cifar100_data_generator.X_valid, cifar100_data_generator.Y_valid

    num_training_xsamples_per_epoch = cifar100_data_generator.X_train.shape[0] // batch_size
    num_validation_xsamples_per_epoch = cifar100_data_generator.X_valid.shape[0] // batch_size

    encoder = c100_classifier.get_layer('embedding')

    c100_classifier.compile(optimizer='sgd', loss=distance_loss(encoder), metrics=['accuracy'])
    c100_classifier.fit_generator(my_generator,
                                  steps_per_epoch=num_training_xsamples_per_epoch,
                                  callbacks=my_callbacks,
                                  validation_data=validation_data,
                                  validation_steps=num_validation_xsamples_per_epoch,
                                  workers=1,
                                  use_multiprocessing=False)

