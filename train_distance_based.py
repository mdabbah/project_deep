import distance_classifier
import cifar100_data_generator as data_genetator
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Nadam
# from keras.applications.vgg16 import VGG16


def distance_loss(encodeing_layer):

    def loss(y_true, y_pred):


        #  concept code
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
        # calculate the embeddings distance from each other in the current batch
        # norms = tf.norm(K.repeat(embeddings, batch_size) - tf.transpose(K.repeat(embeddings, batch_size), [1, 0, 2]), axis=2)
        norms = tf.reduce_sum(tf.squared_difference(K.expand_dims(embeddings, 0) , tf.expand_dims(embeddings, 1)), axis=2)

        # the original classification loss
        total_loss = K.categorical_crossentropy(y_true, y_pred)
        total_loss = tf.reduce_mean(total_loss)
        print_op_pred = tf.print("classification loss: ", total_loss)  # print it

        # boolean matrix s.t. in entry (i,j) is 1 iff label_i == label_j
        y_eq = tf.matmul(y_true, tf.transpose(y_true))
        num_pairs = batch_size**2
        print_op = tf.print("eq. pairs percentage: ", tf.reduce_sum(y_eq)/num_pairs)  # print how manny pairs are equal

        # the proposed distance loss
        distance_loss_eq = tf.reduce_sum(tf.boolean_mask(norms, y_eq))/num_pairs  # loss for pairs with the same label
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
    weights_file = './results/distance_classifiers/' \
                   'distance_classifier_{epoch: 03d}_{val_acc:.3f}_{val_loss:.3f}_{acc:.3f}_{loss:.3f}.h5'
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger('distance_classifier-CIFAR-10.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                       save_weights_only=True, mode='auto')
    my_callbacks = [csv_logger, model_checkpoint] #, lr_reducer, early_stopper]

    # training constants
    batch_size = 100
    distance_margin = 25
    distance_loss_coeff = 0.2
    shuffle = True

    # my__training_generator = data_genetator.datagen.flow(data_genetator.X_train,
    #                                                      np.repeat(np.expand_dims(data_genetator.Y_train, 1), repeats=2,
    #                                                                axis=1),
    #                                                      batch_size=batch_size, shuffle=False)

    my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=shuffle)
    my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=shuffle)

    my_classifier = distance_classifier.DistanceClassifier((32, 32, 3), num_classes=100)

    num_training_xsamples_per_epoch = data_genetator.X_train.shape[0] // batch_size
    num_validation_xsamples_per_epoch = data_genetator.X_valid.shape[0] // batch_size

    encoder = my_classifier.get_layer('embedding')
    optimizer = Nadam(lr=1e-4, clipnorm=1)
    my_classifier.compile(optimizer='Nadam', loss=K.categorical_crossentropy, metrics=['accuracy'])
    my_classifier.fit_generator(my_training_generator,
                                epochs=180,
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

