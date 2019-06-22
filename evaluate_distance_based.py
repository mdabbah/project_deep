from keras import Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from train_distance_based import distance_loss
import distance_classifier
import cifar100_data_generator as data_genetator
import os.path


def distance_score(x_embeddings_test, x_embeddings_train, y_true_train, K=50):
    """
    this is the D(x) described in Amit Mandilbaum and Daphna Winshal's paper
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param x_embeddings_test: test samples embedded
    :param x_embeddings_train: the embedding defined by the neural network for x (matrix num_samples X embedding_size )
    :param y_true_train: hot one encoding for the labels for each given sample (matrix num_samples X num_classes )
    :return: a vector foreach embedding based on the distance confidence score (matrix num_samples X num_classes )
    """
    num_samples = x_embeddings_test.shape[0]
    num_classes = y_true_train.shape[1]
    y_test_confidence = []
    for i in range(num_samples):
        sample_embedding = x_embeddings_test[i]
        distances = np.square(sample_embedding - x_embeddings_train).sum(axis=-1)
        K_nn = np.argsort(distances)[:K]
        K_nn_distances = np.exp(-np.sqrt(distances[K_nn]))
        K_nn_labels = y_true_train[K_nn, :]

        class_indicators = np.eye(num_classes)
        classes_masks = np.matmul(class_indicators, np.transpose(K_nn_labels))

        # foreach class we mask away the samples in Knn that belong to other classes
        class_samples_distances = classes_masks * np.expand_dims(K_nn_distances, axis=0)  # this gives num_classes X K (100 X 50 matrix)
        sum_distances = np.sum(K_nn_distances)
        D_x = np.sum(class_samples_distances, axis=-1)/sum_distances

        y_test_confidence.append(D_x)

    return np.vstack(y_test_confidence)


def negative_entropy(y_pred):
    """
    given a matrix y_pred (num_samples X num_classes .. probability vector for each sample)
    calculates the negative entropy for each row
    :param y_pred:
    :return: a vector, where cell i has the negative entropy for y_pred[i, :]
    """
    negative_entropy_vec = np.sum(np.nan_to_num(-y_pred*np.log(y_pred)), axis=-1)
    return negative_entropy_vec


def pickle_embeddings(model,  pickle_name):
    """pickles the embeddings outputted by the model from its embedding layer into
    a tuple (x_embed , y) where x_embed is num_train_samples X embedding_length.
    pickle name saved is: f'my_embeddings_{pickle_name}'
    model: must me a keras.Model instance which has a layer called 'embedding' """
    import pickle
    layer_name = 'embedding'
    encoder_model = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)
    my_data_generator = data_genetator.MYGenerator('train', batch_size=100)
    embeddings = encoder_model.predict_generator(my_data_generator)

    with open(f'./embeddings/embeddings_for_{pickle_name}.pkl', 'wb') as pkl_out:
        pickle.dump((embeddings, my_data_generator.labels), pkl_out)


def unpickle_embeddings(pickle_name: str):
    """
    returns the tuple pickled by pickle_embeddings method
     a tuple (x_embed , y) where x_embed is num_train_samples X embedding_length
    :param pickle_name: the name of the embeddings pickle file
    :return:  a tuple (x_embed , y)
    """
    fname = f'./embeddings/embeddings_for_{pickle_name}.pkl'
    if not os.path.isfile(fname):
        return False
    import pickle
    with open(fname, 'rb') as pkl_in:
        (embeddings, labels) = pickle.load(pkl_in)

    return embeddings, labels


if __name__ == '__main__':

    # model_name = './results/distance_classifiers_squared/distance_classifier_ 126_0.627_2.389_0.759_1.609.h5'
    # exp_name = 'distance squared trained model, SR predicted'
    model_name = './results/cifar100_crossentropy_classifiers/distance_classifier_ 154_0.615_2.382_0.709_1.032.h5'
    exp_name = 'reg trained model, SR predicted'
    # model_name = './results/distance_classifiers/distance_classifier_ 142_0.640_2.538_0.785_1.719.h5'
    # exp_name = 'distance trained model, SR predicted, confidence '

    #  'max margin'  , 'distance' , 'negative entropy'
    confidence_score = 'distance'

    # loading and compiling the model
    batch_size = 100
    my_classifier = distance_classifier.DistanceClassifier((32, 32, 3), num_classes=100)
    my_classifier.load_weights(model_name)
    optimizer = SGD(lr=1e-2, momentum=0.9)
    encoder = my_classifier.get_layer('embedding')
    my_classifier.compile(optimizer=optimizer, loss=distance_loss(encoder), metrics=['accuracy'])

    # checking predictions
    test_generator = data_genetator.MYGenerator(data_type='test', batch_size=batch_size, shuffle=True)
    y_pred = my_classifier.predict_generator(test_generator)
    y_true = test_generator.labels
    predictions_masks = np_utils.to_categorical(y_pred.argmax(axis=-1))

    if confidence_score == 'max margin':
        y_pred = predictions_masks*y_pred
    elif confidence_score == 'negative entropy':
        y_pred = predictions_masks*np.expand_dims(negative_entropy(y_pred), axis=1)
    elif confidence_score == 'distance':
        model_name = os.path.split(model_name)[-1][:-3]
        pkl = unpickle_embeddings(model_name)
        if not pkl:
            pickle_embeddings(my_classifier, model_name)
            pkl = unpickle_embeddings(model_name)

        layer_name = 'embedding'
        encoder_model = Model(inputs=my_classifier.input,
                              outputs=my_classifier.get_layer(layer_name).output)
        test_embeddings = encoder_model.predict_generator(test_generator)
        distance_scores = distance_score(test_embeddings, x_embeddings_train=pkl[0], y_true_train=pkl[1], K=50)
        y_pred = predictions_masks*distance_scores
        y_pred[np.isnan(y_pred)] = 1e-2

    print(classification_report(y_true, predictions_masks))
    auc = roc_auc_score(y_true, y_pred)
    print(f'auc score for {exp_name} is {auc}')

    loss , acc = my_classifier.evaluate_generator(test_generator)
    print(f'for {exp_name} loss = {loss}, acc = {acc}')
