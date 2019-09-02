import os
from risk_evaluation import distance_predictions, MC_dropout_predictions
import matplotlib.pyplot as plt
from data_generators.facial_keypoints_data_generator import MYGenerator as faical_dg
from data_generators.mnist_data_generator import MYGenerator as mnist_dg
from models.facial_keypoints_arc import FacialKeypointsArc as model
import numpy as np
import keras.backend as K

batch_size = 32
threshold = 0.00045725  # threshold at full coverage calculated from prev. results on validation set used  to calibrate
                      # the rejector min dist  0.252106, MC dropout  0.00045725
uncertainty_criteria = 'min dist'  # MC dropout ,

facial_training_generator = faical_dg('train', use_nans=True)
facial_test_generator = faical_dg('test', batch_size, shuffle=True, use_nans=True,
                                  horizontal_flip_prob=0)

mnist_test_generator = mnist_dg('test', augment=False)
mnist_test_generator.resize(h_factor=96/28, w_factor=96/28, num_channels=3)


input_size = 96, 96, 3
mc_dropout_rate = K.variable(0.)
my_regressor = model(input_size, num_targets=30, num_last_hidden_units=480, mc_dropout_rate=mc_dropout_rate)
regressor_weights_path = r'./results/regression/MSE_updateds_facial_keypoints_arc_facial_key_points/' \
                         r'MSE_updated_facial_key_points_arc_ 478_0.003_0.002_0.00160_ 0.00296.h5'
# regressor_weights_path = r'./results/regression/distance_by_x_encodings_facial_key_points_arc_facial_key_points/' \
#                          r'distance_by_x_encoding_facial_key_points_arc_ 440_0.003_0.002_0.00170_ 0.00293.h5'

# load weights
my_regressor.name = os.path.split(regressor_weights_path)[-1][:-3]
my_regressor.load_weights(regressor_weights_path)

if uncertainty_criteria == 'min dist':

    _, uncertainty_per_facial_test_sample = distance_predictions(my_regressor,
                                                                               facial_test_generator,
                                                                               facial_training_generator)

    _, uncertainty_per_facial_training_sample = distance_predictions(my_regressor, facial_training_generator,
                                                                               facial_training_generator)

    _, uncertainty_per_mnist_test_sample = distance_predictions(my_regressor, mnist_test_generator,
                                                                           facial_training_generator)
else:  # MC dropout

    _, uncertainty_per_facial_test_sample = MC_dropout_predictions(my_regressor, facial_test_generator, mc_dropout_rate)

    _, uncertainty_per_facial_training_sample = MC_dropout_predictions(my_regressor, facial_training_generator, mc_dropout_rate)

    _, uncertainty_per_mnist_test_sample = MC_dropout_predictions(my_regressor, mnist_test_generator, mc_dropout_rate)


print(f"overlap percentage "
      f"{np.sum(uncertainty_per_mnist_test_sample < threshold) / len(uncertainty_per_mnist_test_sample)}")

order = 0
number_closest_to_face = uncertainty_per_mnist_test_sample.argsort()[0+order]
face_farthest_from_faces = uncertainty_per_facial_test_sample.argsort()[-1-order]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mnist_test_generator.imgs[number_closest_to_face])
plt.title(f'number closest to face, \n min dist.{uncertainty_per_mnist_test_sample[number_closest_to_face] :.5f}')

plt.subplot(1, 2, 2)
plt.imshow(facial_test_generator.images[face_farthest_from_faces])
plt.title(f'number closest to face, \n min dist. {uncertainty_per_facial_test_sample[face_farthest_from_faces] :.5f}')


def plot_shared_hist(hist1_data, hist2_data, bin_max=None, normalize=True, labels=None):
    plt.figure()
    bins = None
    if isinstance(bin_max, float):
        bins = np.linspace(0, bin_max, 100)
    x1, _, p1 = plt.hist(hist1_data, bins, density=True, label=labels[0],
                         fc=(1, 0, 0, 0.5))
    x2, _, p2 = plt.hist(hist2_data, bins, density=True, label=labels[1], fc=(0, 0, 1, 0.5))
    plt.legend(loc='upper right')
    plt.title(f'{uncertainty_criteria} criteria on facial ds regressor\n normalized histogram')

    if normalize:
        heights = []
        for item1, item2 in zip(p1, p2):
            h1 = item1.get_height() / sum(x1)
            h2 = item2.get_height() / sum(x2)
            heights.extend([h1, h2])
            item1.set_height(h1)
            item2.set_height(h2)
        plt.ylim([0, np.max(heights) * 1.1])


bin_max = 0.8
plot_shared_hist(uncertainty_per_facial_training_sample, uncertainty_per_facial_test_sample,
                 labels=['training facial', 'test facial'], bin_max=bin_max)
plot_shared_hist(uncertainty_per_facial_training_sample, uncertainty_per_mnist_test_sample,
                 labels=['training facial', 'test mnist'], bin_max=bin_max)
plot_shared_hist(uncertainty_per_facial_test_sample, uncertainty_per_mnist_test_sample,
                 labels=['test facial', 'test mnist'], bin_max=bin_max)

