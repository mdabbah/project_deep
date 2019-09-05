import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from models.SVHN_arch_classifier import DistanceClassifier as svhn_model
from data_generators.svhn_data_generator import MYGenerator
from sklearn.manifold import TSNE
from keras import Model


test_generator = MYGenerator('test')
my_model = svhn_model()
model_name = './results/distance_classifiers_svhn_SVHN_new/distance_classifier_svhn_ 24_0.921_0.895_0.936_0.850.h5'
my_model.load_weights(model_name)
my_encoder = Model(my_model.input, my_model.get_layer('embedding').output)
test_embeddings = my_encoder.predict_generator(test_generator)
true_labels = test_generator.gt.T == 1

tsne_projector = TSNE(verbose=1)
test_projected = tsne_projector.fit_transform(test_embeddings)
cmap = cm.rainbow(np.linspace(0.0, 1.0, 10))

for concept in range(test_generator.gt.shape[1]):
    concept_mask = true_labels[concept, :]
    plt.scatter(test_projected[concept_mask, 0], test_projected[concept_mask, 1], label='number' + str(concept), c=cmap[concept])


plt.title('t-sne projection of svhn test encodings \nwith distance trained model')
plt.legend()
plt.show()
