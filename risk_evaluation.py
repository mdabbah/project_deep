import os

from keras import Model, Input
import numpy as np
from keras.layers import Dropout, Dense, Lambda

from evaluate_distance_based import pickle_embeddings, unpickle_embeddings
from train_distance_based_regression import MSE_updated
import tensorflow as tf
from keras.backend import eval
import keras.backend as K


def calc_selective_risk(coverage: float, test_losses, uncertainty_on_validation, uncertainty_on_test):
    """
    given the losses on test dataset,  the desired coverage
    uncertainty scores on validation and uncertainty scores on test
    we calculate the selective risk which is defined as:
    selective_risk = E{loss(y_hat_i, y_i)*(uncertainty_i< TH)}/E{uncertainty_i< TH}
    TH is the threshold that is the percitile 'coverage' of uncertanty on validation
    see selective risk in "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"
    paper
    :param coverage: float [0,1]
    :param test_losses: vector of size z X num_test_samples where in cell i is the loss on sample i
    :param uncertainty_on_validation: vector of size z X num_validation_samples where in cell i is the uncertainty on
     sample i
    :param uncertainty_on_test: vector of size z X num_test_samples where in cell i is the uncertainty on
     sample i
    :return: selective risk
    """
    th = np.percentile(uncertainty_on_validation, coverage * 100)

    loss_on_covered = np.mean(test_losses[uncertainty_on_test < th])
    test_coverage = np.mean(uncertainty_on_test < th)
    risk = loss_on_covered / test_coverage
    print(f'for given coverage: ,{coverage}, threshold is: ,{th :.6f},'
          f' test coverage: ,{test_coverage:.6f}, '
          f'test loss on covered ,{loss_on_covered :.6f}, '
          f'risk is: ,{risk :.6f}')

    return np.mean(test_losses[uncertainty_on_test < th])


def turn_on_dropout(model: Model, new_rate = 0.05):
    """
    since keras doesn't have the option to turn on dropout on testing,
    my workaround is to replace the dropout layers which lambda layers that use dropout function
    that way it is always turned on
    WARNING: works only on linear/sequential networks
    :param model: model to replace the dropout layers in
    :return: the updated model
    """

    x = input =Input(shape=model.input_shape[1:])
    for layer in model.layers:
        if layer.name.startswith('drop_out_to_turn_on'):
            print('found drop_out_to_turn_on layer ')
            x = Lambda(lambda l_in: K.dropout(l_in, level=new_rate), name='drop_out_to_turn_on')(x)
            continue
        x = layer(x)

    new_model = Model(inputs=input, outputs=x)
    return new_model


def MC_dropout_predictions(my_regressor: Model, test_generator, num_evaluations: int = 200):
    """
    replacing the model with a new model where its dropouts are turned on during testing as well
    in order to preform MC dropout evaluation
    for each sample in testing generator we apply the new model #num_evaluation times
    our prediction will be the mean value of all evaluations
    and our uncertainty will be the mean(std)
    :param my_regressor: regression model
    :param test_generator: tersting data generator
    :param num_evaluations: # evaluations
    :return: a tuple (predictions, uncertainty)
    """

    predictions = np.zeros((num_evaluations, *test_generator.gt.shape))
    # old way
    # model_dropout_turned_on = turn_on_dropout(my_regressor)
    # new way
    K.set_value(mc_dropout_rate, value=0.05)
    model_dropout_turned_on = my_regressor

    for i in range(num_evaluations):
        predictions[i] = np.squeeze(model_dropout_turned_on.predict_generator(test_generator))

    uncertainty_per_test_sample = np.var(predictions, axis=0)
    if len(uncertainty_per_test_sample.shape) > 1:
        # if we have more than one target per sample,
        # our uncertainty in this test sample is the avg uncertainty for its targets
        uncertainty_per_test_sample = np.mean(uncertainty_per_test_sample, axis=-1)
    return np.mean(predictions, axis=0), uncertainty_per_test_sample


def distance_predictions(my_regressor, test_generator, training_generator):
    """
    the uncertainty metric I suggested to contend with MC dropout.
    this method returns the predictions on the test genertor and the uncertanty score
    the uncertanty score for a test sample x is the distance squared from the nearest
    training sample in the embedding space
    the embedding is the output of the embedding layer in the regressor
    :param my_regressor: a nn model for the problem, it should have a layer called 'embedding'
    :param test_generator: a generator for the test samples
    :param training_generator: a generator for the training samples
    :return: predictions_on_testset, uncertainty_per_test_sample
    """

    # get training embeddings
    dataset_name = 'regression/' + test_generator.dataset_name
    model_name = my_regressor.name
    pkl = unpickle_embeddings(model_name, dataset_name)
    if not pkl:
        pickle_embeddings(my_regressor, model_name, dataset_name, training_generator)
        pkl = unpickle_embeddings(model_name, dataset_name)

    training_embeddings, training_targets = pkl

    #  get teat embeddings
    layer_name = 'embedding'
    encoder_model = Model(inputs=my_regressor.input,
                          outputs=my_regressor.get_layer(layer_name).output)
    test_embeddings = encoder_model.predict_generator(test_generator)

    # calculate the distances between each test sample and all training samples in embedding space
    batch_size = 32  # done in "batches" because of memory constrains
    num_test_samples = test_embeddings.shape[0]

    dists = np.zeros((test_embeddings.shape[0], training_embeddings.shape[0]))
    for bt in range(np.int(np.ceil(num_test_samples//batch_size))+1):
        test_embeddings_bt = test_embeddings[bt*batch_size:batch_size*(bt+1), :]
        if test_embeddings_bt.size == 0:
            break
        batch_dists = np.sum(np.square(np.expand_dims(training_embeddings, 0) - np.expand_dims(test_embeddings_bt, 1)), axis=-1)
        dists[batch_size*bt:batch_size*(bt+1), :] = batch_dists

    predictions_on_test = my_regressor.predict_generator(test_generator)
    uncertainty_per_test_sample = np.min(dists, axis=1)
    return predictions_on_test, uncertainty_per_test_sample


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred.round())


if __name__ == '__main__':

    # general params
    data_set = 'mnist'  # concrete_strength ,  facial_key_points , mnist
    uncertainty_metric = 'min_distance'  # options 'min_distance' , 'MC_dropout_std'
    batch_size = 32
    input_size = None
    mc_dropout_rate = K.variable(value=0)
    loss_function = lambda y,  y_hat: eval(MSE_updated(y, y_hat, return_vec=True))
    my_regressor = None
    training_generator = None
    validation_generator = None
    test_generator = None
    regressor_weights_path = None

    # loading data
    if data_set == 'facial_key_points':
        from data_generators import facial_keypoints_data_generator as data_genetator

        training_generator = data_genetator.MYGenerator('train', use_nans=True)
        validation_generator = data_genetator.MYGenerator('valid', batch_size, shuffle=True, use_nans=True,
                                                          horizontal_flip_prob=0)
        test_generator = data_genetator.MYGenerator('test', batch_size, shuffle=True, use_nans=True,
                                                    horizontal_flip_prob=0)

        from models.facial_keypoints_arc import FacialKeypointsArc as model
        input_size = 96, 96, 3
        my_regressor = model(input_size, num_targets=30, num_last_hidden_units=480, mc_dropout_rate=mc_dropout_rate)
        regressor_weights_path = r'./results/regression/MSE_updateds_facial_keypoints_arc_facial_key_points/' \
                                 r'MSE_updated_facial_key_points_arc_ 478_0.003_0.002_0.00160_ 0.00296.h5'
        encoder_weights_path = regressor_weights_path

        # loss for evaluation is RMSE*48 (times 48 for rescaling to original problem values)
        loss_function = lambda y,  y_hat: np.sqrt(eval(MSE_updated(y, y_hat, return_vec=True))) * 48

        print("evaluating on facial keypoints")
    elif data_set == 'fingers':
        raise ValueError("not supported yet")
    elif data_set == 'concrete_strength':
        from data_generators import concrete_dataset_generator as data_genetator
        from models.concrete_strength_arc import simple_FCN as model
        training_generator = data_genetator.MYGenerator('train')
        validation_generator = data_genetator.MYGenerator('valid', batch_size, True)
        test_generator = data_genetator.MYGenerator('test', batch_size, True)
        input_size = 8
        my_regressor = model(input_size, 1, mc_dropout_rate=mc_dropout_rate)
        regressor_weights_path = r'./results/regression/l1_smooth_losss_simple_FCN_concrete_strength/' \
                       r'l1_smooth_loss_simple_FCN_ 659_3.177_3.137_27.02343_ 24.50994.h5'
        # regressor_weights_path = r'./results/regression/distance_by_x_encodings_simple_FCN_concrete_strength/' \
        #                r'distance_by_x_encoding_simple_FCN_ 573_32.042_31.230_26.25158_ 27.36713.h5'
    if data_set == 'mnist':
        from data_generators import mnist_data_generator as data_genetator
        from models.mnist_simple_arc import mnist_simple_arc as model

        training_generator = data_genetator.MYGenerator('train')
        validation_generator = data_genetator.MYGenerator('valid', batch_size, augment=True)
        test_generator = data_genetator.MYGenerator('test', batch_size, augment=True)
        input_size = 28, 28, 1
        my_regressor = model(input_size, 1, mc_dropout_rate=mc_dropout_rate)
        regressor_weights_path = r'./results/regression/distance_by_x_encodings_simple_CNN_mnist/' \
                                 r'distance_by_x_encoding_simple_CNN_ 45_0.114_0.304_0.28383_ 0.104280.72587_ 0.98250.h5'

        loss_function = accuracy

    # load weights
    my_regressor.name = os.path.split(regressor_weights_path)[-1][:-3]
    my_regressor.load_weights(regressor_weights_path)

    #  choose the uncertainty metric to evaluate
    if uncertainty_metric == 'MC_dropout_std':
        valid_predictions, valid_uncertainty = MC_dropout_predictions(my_regressor, validation_generator)
        test_predictions, test_uncertainty = MC_dropout_predictions(my_regressor, test_generator)
    else:
        valid_predictions, valid_uncertainty = distance_predictions(my_regressor, validation_generator, training_generator)
        test_predictions, test_uncertainty = distance_predictions(my_regressor, test_generator, training_generator)

    y_true = test_generator.gt if len(test_generator.gt.shape) > 1 else np.expand_dims(test_generator.gt, 1)
    y_pred = test_predictions if len(test_predictions.shape) > 1 else np.expand_dims(test_predictions, 1)
    test_losses = loss_function(y_true, y_pred)

    # validation_MSE = eval(MSE_updated(validation_generator.gt, valid_predictions))
    # test_MSE = eval(MSE_updated(test_generator.gt, test_predictions))
    # print(f'valid MSE is {validation_MSE}, test MSE is {test_MSE}')

    coverages = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.]
    risk_coverages = np.zeros(len(coverages))
    for i, coverage in enumerate(coverages):
        risk_coverages[i] = calc_selective_risk(coverage=coverage, test_losses=test_losses,
                                                uncertainty_on_validation=valid_uncertainty,
                                                uncertainty_on_test=test_uncertainty)

    [print(str(c)) for c in risk_coverages]


