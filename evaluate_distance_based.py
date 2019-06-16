from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report, roc_auc_score

from train_distance_based import distance_loss
import distance_classifier
import cifar100_data_generator as data_genetator


def distance_score(x_embedding):
    """
    this is the D(x) described in Amit Mandilbaum and Daphna Winshal's paper
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param x_embedding: the embedding defined by the neural network for x
    :return: a vector foreach y
    """
    pass


def pickle_embeddings(model, x_train, y_train, pickle_name):
    """pickles the embeddings outputted by the model from its embedding layer into
    a tuple (x_embed , y) where x_embed is num_train_samples X embedding_length"""
    pass


if __name__ == '__main__':

    model_name = './results/distance_classifiers/distance_classifier_ 126_0.627_2.389_0.759_1.609.h5'
    exp_name = 'distance trained model, SR predicted'
    batch_size = 100
    my_classifier = distance_classifier.DistanceClassifier((32, 32, 3), num_classes=100)
    my_classifier.load_weights(model_name)
    optimizer = SGD(lr=1e-2, momentum=0.9)
    encoder = my_classifier.get_layer('embedding')
    my_classifier.compile(optimizer=optimizer, loss=distance_loss(encoder), metrics=['accuracy'])

    test_generator = data_genetator.MYGenerator(data_type='test', batch_size=batch_size, shuffle=True)
    y_pred = my_classifier.predict_generator(test_generator)
    y_true = test_generator.labels
    print(classification_report(y_true, y_pred))

    auc = roc_auc_score(y_true, y_pred)
    print(f'auc score for {exp_name}')
