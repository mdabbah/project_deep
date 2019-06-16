from keras.models import load_model



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
    model.pr


if __name__ == '__main__':

    model_name = './results/distance_classifiers/distance_classifier_ 126_0.627_2.389_0.759_1.609.h5'
    model = load_model(model_name)