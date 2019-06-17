from keras import Model
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from train_distance_based import distance_loss
import distance_classifier
import cifar100_data_generator as data_genetator


def distance_score(x_embeddings, y_true, K=50):
    """
    this is the D(x) described in Amit Mandilbaum and Daphna Winshal's paper
    "Distance-based Confidence Score for Neural Network Classifiers"
    https://arxiv.org/abs/1709.09844
    :param x_embeddings: the embedding defined by the neural network for x (matrix num_samples X embedding_size )
    :param y_true: hot one encoding for the labels for each given sample (matrix num_samples X num_classes )
    :return: a vector foreach embedding based on the distance confidence score (matrix num_samples X num_classes )
    """
    num_samples = x_embeddings.shape[0]
    num_classes = y_true.shape[1]
    new_y_pred = []
    for i in range(num_samples):
        sample_embedding = x_embeddings[i]
        distances = np.square(sample_embedding - x_embeddings).sum(axis=-1)
        K_nn = np.argsort(distances)[1:K]
        K_nn_distances = np.exp(-distances[K_nn])
        K_nn_labels = y_true[K_nn, :]

        class_indicators = np.eye(num_classes)
        classes_masks = np.matmul(class_indicators, np.transpose(K_nn_labels))

        # foreach class we mask away the samples in Knn that belong to other classes
        class_samples_distances = classes_masks * np.expand_dims(K_nn_distances, axis=0)  # this gives num_classes X K (100 X 50 matrix)

        D_x = np.sum(class_samples_distances, axis=-1)/np.sum(K_nn_distances)

        new_y_pred.append(D_x)



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

    with open(f'./embeddings/embeddings_for_{pickle_name}', 'wb') as pkl_out:
        pickle.dump((embeddings, my_data_generator.labels), pkl_out)


def unpickle_embeddings(pickle_name: str):
    """
    returns the tuple pickled by pickle_embeddings method
     a tuple (x_embed , y) where x_embed is num_train_samples X embedding_length
    :param pickle_name: the name of the embeddings pickle file
    :return:  a tuple (x_embed , y)
    """
    import pickle
    with open(f'./embeddings/embeddings_for_{pickle_name}', 'rb') as pkl_in:
        (embeddings, labels) = pickle.load(pkl_in)

    return embeddings, labels


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
