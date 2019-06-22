import distance_classifier
import stl10_data_generator as data_genetator
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Nadam, SGD
from keras.applications.vgg16 import VGG16


lr_cache_cifar100 = {67: 5e-3, 112: 5e-4, 145: 5e-5}
lr_cache_stl10 = {134: 5e-3, 223: 5e-4, 289: 5e-5}


def lr_scheduler_maker(dataset_name: str, scheduling_dictionary: dict =None):
    """
    takes a dataset name and returns an lr_scheduler function
    :param dataset_name: name of the dataset
    :param scheduling_dictionary: should be a dictionarty where key is epoch : value is lr
    you can also just define the transitional epochs in this dictionary
    :return: lr_scheduler function to be passed to LearningRateScheduler object
    """
    if dataset_name == 'cifar100':
        lr_cache = lr_cache_cifar100
    elif dataset_name == 'stl10':
        lr_cache = lr_cache_stl10
    else:
        Warning('supported datasets are cifar100, and stl10, will be using'
                ' the scheduling_dictionary ')
        if scheduling_dictionary is None:
            ValueError('bad parameters: no scheduling dict was passed')
        lr_cache = scheduling_dictionary

    def lr_scheduler(epoch, current_lr):
        """
        the function used to decrease the learning rate as suggested by
        "Distance-based Confidence Score for Neural Network Classifiers"
        https://arxiv.org/abs/1709.09844
        :param epoch: the current epoch
        :param current_lr: the current learning rate
        :return: the new learning rate
        """

        new_lr = current_lr

        return lr_cache.get(epoch, new_lr)

    return lr_scheduler


def distance_loss(encodeing_layer, batch_size=100, distance_margin = 25, distance_loss_coeff = 0.2):
    """
    the loss function that depends on the encoding of the second to last layer
    as suggested by
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param encodeing_layer: which layer we want to use as encoder
    :return: loss function
    """
    def loss(y_true, y_pred):


        #  concept code?
        # y_true = tf.argmax(y_true, axis=1)
        # loss = 0
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         embedd_i = embeddings[i, :]
        #         embedd_j = embeddings[j, :]
        #         distance = tf.norm(embedd_i - embedd_j)
        #         if y_true[i] == y_true[j]:
        #             loss += distance
        #         else:
        #             loss += tf.maximum(0., distance_margin - distance)
        #
        # return loss*distance_loss_coeff/10000

        # print the embeddings
        # print_op_embedd = tf.print("embeddings: ", embeddings)

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
        num_pairs = batch_size**2
        print_op = tf.print("eq. pairs percentage: ", tf.reduce_sum(y_eq)/num_pairs)  # print how manny pairs are equal

        # the proposed distance loss
        distance_loss_eq = tf.reduce_sum(tf.boolean_mask(norms, y_eq - tf.eye(batch_size)))/num_pairs  # loss for pairs with the same label
        distance_loss_diff = tf.reduce_sum(tf.maximum(0., distance_margin - tf.boolean_mask(norms, 1-y_eq)))/num_pairs  # loss for pairs with different label
        # print them
        print_op2 = tf.print("loss equals: ", distance_loss_eq)
        print_op3 = tf.print("loss diff: ", distance_loss_diff)

        total_loss += (distance_loss_eq + distance_loss_diff) * distance_loss_coeff
        with tf.control_dependencies([print_op, print_op2, print_op3]):
            return total_loss

    return loss


if __name__ == '__main__':

    # file name & after epoch callbacks
    weights_file = './results/distance_classifiers_stl10/' \
                   'distance_classifier_{epoch: 03d}_{val_acc:.3f}_{val_loss:.3f}_{acc:.3f}_{loss:.3f}.h5'
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler_maker('stl10'))
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger('distance_classifier-stl-10.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                       save_weights_only=True, mode='auto')
    my_callbacks = [csv_logger, model_checkpoint, lr_scheduler_callback] # lr_scheduler_callback , lr_reducer, early_stopper]

    # training constants
    batch_size = 100
    distance_margin = 25
    distance_loss_coeff = 0.2
    nb_classes = 10
    shuffle = False
    input_size = (96, 96, 3)  # for cifar100 = (32, 32, 3)
    num_epochs = 360  # for cifar100 180

    my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=shuffle)
    my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=shuffle)

    my_classifier = distance_classifier.DistanceClassifier(input_size, num_classes=nb_classes)

    num_training_xsamples_per_epoch = data_genetator.Y_train.shape[0] // batch_size
    num_validation_xsamples_per_epoch = data_genetator.Y_valid.shape[0] // batch_size

    encoder = my_classifier.get_layer('embedding')
    optimizer = SGD(lr=1e-2, momentum=0.9) #Nadam(lr=1e-4, clipnorm=1)
    my_classifier.compile(optimizer=optimizer, loss=distance_loss(encoder, batch_size), metrics=['accuracy'])
    my_classifier.fit_generator(my_training_generator,
                                epochs=num_epochs,
                                steps_per_epoch=num_training_xsamples_per_epoch,
                                callbacks=my_callbacks,
                                validation_data=my_validation_generator,
                                validation_steps=num_validation_xsamples_per_epoch,
                                workers=1,
                                use_multiprocessing=False)

    test_generator = data_genetator.MYGenerator(data_type='test', batch_size=batch_size, shuffle=True)

    loss, acc = my_classifier.evaluate_generator(test_generator,
                                                    steps=data_genetator.X_test.shape[0]//batch_size)

    print(f'test acc {acc}, test loss {loss}')

