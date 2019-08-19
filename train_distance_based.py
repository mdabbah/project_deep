import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import keras.applications.inception_resnet_v2 as inception_resnet_v2
from keras_contrib.applications.resnet import ResNet18
import os


lr_cache_cifar100 = {67: 5e-3, 112: 5e-4, 145: 5e-5}
lr_cache_stl10 = {134: 5e-3, 223: 5e-4, 289: 5e-5}
lr_cache_svhn = {0: 1e-3, 9: 5e-4, 15: 5e-5}


def lr_scheduler_maker(dataset_name: str, scheduling_dictionary: dict =None):
    """
    takes a dataset name and returns an lr_scheduler function
    :param dataset_name: name of the dataset
    :param scheduling_dictionary: should be a dictionarty where key is epoch : value is lr
    you can also just define the transitional epochs in this dictionary
    :return: lr_scheduler function to be passed to LearningRateScheduler object
    """
    if dataset_name.startswith('CIFAR-10'):
        lr_cache = lr_cache_cifar100
    elif dataset_name == 'stl10':
        lr_cache = lr_cache_stl10
    elif data_set == 'SVHN':
        lr_cache = lr_cache_svhn
    else:
        Warning('supported datasets are cifar100, and stl10, will be using'
                ' the scheduling_dictionary ')
        if scheduling_dictionary is None:
            raise ValueError('bad parameters: no scheduling dict was passed')
        lr_cache = scheduling_dictionary
    print(f'chosen lr scheduling is:{lr_cache}')

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


def build_inception_resnet_classifier(input_size=(96, 96, 3)):
    """
    takes the base inception resnet arch cuts the top off and adds 2 more dense layers
    one for embedding and one for the final classification
    :param input_size: the input size of the network
    :return: a keras Model
    """
    num_classes = data_genetator.nb_classes
    base_model = inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=input_size,
                                                       classes=num_classes, pooling='avg')

    x = base_model.output
    # let's add 2 fully-connected layer for embedding
    x = Dense(512, activation='relu')(x)
    x = Dense(300, activation='relu', name='embedding')(x)
    # x = Flatten(name='embedding')(x)
    # and a prediction layer
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-10]:
        layer.trainable = False

    return model


if __name__ == '__main__':

    # wheere to save weights , dataset & training details change if needed
    data_set = 'CIFAR-100'
    training_type = 'crossentropy_classifier'  # options 'crossentropy_classifier', 'distance_classifier'
    arch = 'ResNet18'
    weights_folder = f'./results/{training_type}s_{arch}_{data_set}'
    os.makedirs(weights_folder, exist_ok=True)
    weights_file = f'{weights_folder}/{training_type}_{arch}' \
                   '_{epoch: 03d}_{val_acc:.3f}_{val_loss:.3f}_{acc:.3f}_{loss:.3f}.h5'

    # callbacks change if needed
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler_maker(data_set))
    early_stopper = EarlyStopping(min_delta=0.001, patience=20)
    csv_logger = CSVLogger(f'{training_type}-{data_set}.csv')
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                       save_weights_only=True, mode='auto')
    my_callbacks = [csv_logger, model_checkpoint, lr_scheduler_callback] # lr_scheduler_callback , lr_reducer, early_stopper]

    # loading data
    if data_set == 'CIFAR-10':
        from data_generators import cifar10_data_generator as data_genetator, svhn_data_generator as data_genetator, \
            cifar100_data_generator as data_genetator

        print("training on cifar10")
    elif data_set == 'CIFAR-100':
        print("training on cifar100")
    elif data_set == 'SVHN':
        print("training on SVHN")

    # training constants, change if needed
    num_epochs = 180
    batch_size = 100
    distance_margin = 25
    distance_loss_coeff = 0.2
    shuffle = False
    input_size = (32, 32, 3)
    num_training_xsamples_per_epoch = data_genetator.X_train.shape[0] // batch_size
    num_validation_xsamples_per_epoch = data_genetator.X_valid.shape[0] // batch_size
    my_classifier = None
    preprocess_input_fun = None

    # choosing arch and optimizer
    if data_set.startswith('SVHN'):
        optimizer = Adam(lr=1e-3)
        num_epochs = 26
        print('SVHN suggested arch chosen, optimizer adam')
    else:
        if arch == 'ResNet18':
            my_classifier = ResNet18(input_size, data_genetator.nb_classes)
            my_classifier.layers[-3].name = 'embedding'
            print('chose resnet18v2 arch')
            # optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
            num_epochs = 200
            my_callbacks[-1] = LearningRateScheduler(lr_scheduler_maker('', scheduling_dictionary={60: 0.1/5, 120: 0.1/25, 180: 0.1/125}))
            del my_callbacks[-1]
            optimizer = 'adam'
            shuffle = True
        elif arch == 'inception_resnet_v2':
            input_size = (96, 96, 3)
            my_classifier = build_inception_resnet_classifier(input_size)
            preprocess_input_fun = inception_resnet_v2.preprocess_input
            print('chose inception resent v2 arch')
            optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)

        else:
            from models import distance_classifier, SVHN_arch_classifier as distance_classifier

            print('cifar100 suggested arch chosen, optimizer SGD w. momentum')
            optimizer = SGD(lr=1e-2, momentum=0.9)


    # data generators
    my_training_generator = data_genetator.MYGenerator(data_type='train', batch_size=batch_size, shuffle=shuffle,
                                                       input_size=input_size, preprocessing_fun=preprocess_input_fun)
    my_validation_generator = data_genetator.MYGenerator(data_type='valid', batch_size=batch_size, shuffle=shuffle,
                                                         input_size=input_size, preprocessing_fun=preprocess_input_fun)

    # creating the classification model and compiling it
    if my_classifier is None:
        my_classifier = distance_classifier.DistanceClassifier(input_size, num_classes=data_genetator.nb_classes)

    encoder = my_classifier.get_layer('embedding')
    loss_function = \
        distance_loss(encoder, batch_size) if training_type.startswith('distance') else K.categorical_crossentropy

    my_classifier.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # start training
    my_classifier.fit_generator(my_training_generator,
                                epochs=num_epochs,
                                steps_per_epoch=num_training_xsamples_per_epoch,
                                callbacks=my_callbacks,
                                validation_data=my_validation_generator,
                                validation_steps=num_validation_xsamples_per_epoch,
                                workers=1,
                                use_multiprocessing=0)

    test_generator = data_genetator.MYGenerator(data_type='test', batch_size=batch_size, shuffle=True,
                                                input_size=input_size, preprocessing_fun=preprocess_input_fun)

    # check acc
    loss, acc = my_classifier.evaluate_generator(test_generator,
                                                    steps=data_genetator.X_test.shape[0]//batch_size)

    print(f'test acc {acc}, test loss {loss}')

