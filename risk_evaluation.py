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
    uncertainty_on_validation = np.sort(uncertainty_on_validation)
    th = np.percentile(uncertainty_on_validation, coverage * 100)

    loss_on_covered = np.mean(test_losses[uncertainty_on_test < th])
    test_coverage = np.mean(uncertainty_on_test < th)
    risk = loss_on_covered / test_coverage
    print(f'for given coverage: {coverage}, threshold is: {th},'
          f' test coverage: {test_coverage}, '
          f'test loss on coverd {loss_on_covered}, '
          f'risk is: {risk}')

    return np.mean(test_losses[uncertainty_on_test < th]) / np.mean(uncertainty_on_test < th)


def turn_on_dropout(model: Model, new_rate = 0.05):
    """
    since keras doesn't have the option to turn on dropout on testing,
    my workaround is to replace the dropout layers which lambda layers that use dropout function
    that way it is always turned on
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

    predictions = np.zeros((num_evaluations, *test_generator.labels.shape))
    model_dropout_turned_on = turn_on_dropout(my_regressor)
    for i in range(num_evaluations):
        predictions[i, :] = np.squeeze(model_dropout_turned_on.predict_generator(test_generator))

    uncertainty_per_test_sample = np.var(predictions, axis=0)
    if len(uncertainty_per_test_sample.shape) > 1:
        uncertainty_per_test_sample = np.mean(uncertainty_per_test_sample, axis=-1)
    return np.mean(predictions, axis=0), uncertainty_per_test_sample


def distance_predictions(my_regressor, my_encoder, test_generator, training_generator):

    dataset_name = 'regression/' + test_generator.dataset_name
    model_name = my_regressor.name
    pkl = unpickle_embeddings(model_name, dataset_name)
    if not pkl:
        pickle_embeddings(my_encoder, model_name, dataset_name, training_generator)
        pkl = unpickle_embeddings(model_name, dataset_name)

    training_embeddings, training_targets = pkl
    if len(training_targets.shape) == 1:
        training_targets = np.expand_dims(training_targets, 1)
    layer_name = 'embedding'
    encoder_model = Model(inputs=my_encoder.input,
                          outputs=my_encoder.get_layer(layer_name).output)
    test_embeddings = encoder_model.predict_generator(test_generator)

    batch_size = 32
    num_training_samples = training_embeddings.shape[0]
    num_test_samples = test_embeddings.shape[0]
    embedding_len_per_target = int(training_embeddings.shape[1]//training_targets.shape[1])  # we have y_true.shape[0] targets

    dists = np.zeros((test_embeddings.shape[0], training_embeddings.shape[0]))
    for bt in range(np.int(np.ceil(num_test_samples//batch_size))+1):
        test_embeddings_bt = test_embeddings[bt*batch_size:batch_size*(bt+1), :]
        if test_embeddings_bt.size == 0:
            break
        actual_bt_size = test_embeddings_bt.shape[0]
        mask = np.repeat(np.expand_dims(np.isnan(training_targets), axis=0), actual_bt_size, axis=0)
        batch_dists = np.sum(np.square(np.expand_dims(training_embeddings, 0) - np.expand_dims(test_embeddings_bt, 1)), axis=-1)
        # batch_dists[mask] = np.inf
        dists[batch_size*bt:batch_size*(bt+1), :] = batch_dists

    return my_regressor.predict_generator(test_generator), np.min(dists, axis=1)


if __name__ == '__main__':

    # general params
    data_set = 'facial_key_points'
    loss_type = 'MSE'  # options 'l1_smoothed', 'distance_classifier'
    arch = 'facial_key_points_arc'
    uncertainty_metric = 'min_distance'  # options 'min_distance' , 'MC_dropout_std'
    batch_size = 32
    input_size = None
    loss_function = MSE_updated
    my_regressor = None
    training_generator = None
    validation_generator = None
    test_generator = None
    regressor_weights_path = None
    my_encoder = None
    encoder_weights_path = None

    # loading data
    if data_set == 'facial_key_points':
        from data_generators import facial_keypoints_data_generator as data_genetator

        training_generator = data_genetator.MYGenerator('train', use_nans=True)
        validation_generator = data_genetator.MYGenerator('valid', batch_size, shuffle=True, use_nans=True,
                                                          horizontal_flip_prob=0)
        test_generator = data_genetator.MYGenerator('test', batch_size, shuffle=True, use_nans=True,
                                                    horizontal_flip_prob=0)
        input_size = 96, 96, 3
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
        my_regressor = model(input_size, 1)
        my_encoder = model(input_size, 1, False)
        regressor_weights_path = r'./results/regression/l1_smooth_losss_simple_FCN_concrete_strength/' \
                       r'l1_smooth_loss_simple_FCN_ 659_3.177_3.137_27.02343_ 24.50994.h5'
        encoder_weights_path = r'./results/regression/distance_by_x_encodings_simple_FCN_concrete_strength/' \
                       r'distance_by_x_encoding_simple_FCN_ 778_0.000_0.000.h5'

    # building the model
    if data_set.startswith('facial'):
        from models.facial_keypoints_arc import FacialKeypointsArc as model
        my_regressor = model(input_size, 30, 480)
        regressor_weights_path = r'./results/regression/distance_by_x_encodings_facial_key_points_arc_facial_key_points/' \
                                 r'distance_by_x_encoding_facial_key_points_arc_ 440_0.003_0.002_0.00170_ 0.00293.h5'
        encoder_weights_path = regressor_weights_path
        my_encoder = my_regressor

    # load weights
    my_encoder.name = 'encoder_' + os.path.split(encoder_weights_path)[-1][:-3]
    my_regressor.load_weights(regressor_weights_path)
    my_encoder.load_weights(encoder_weights_path)
    my_encoder = my_regressor

    if uncertainty_metric == 'MC_dropout_std':
        valid_predictions, valid_uncertainty = MC_dropout_predictions(my_regressor, validation_generator)
        test_predictions, test_uncertainty = MC_dropout_predictions(my_regressor, test_generator)
    else:
        valid_predictions, valid_uncertainty = distance_predictions(my_regressor, my_encoder, validation_generator, training_generator)
        test_predictions, test_uncertainty = distance_predictions(my_regressor, my_encoder, test_generator, training_generator)

    y_true = test_generator.labels if len(test_generator.labels.shape) > 1 else np.expand_dims(test_generator.labels, 1)
    y_pred = test_predictions if len(test_predictions.shape) > 1 else np.expand_dims(test_predictions, 1)
    test_losses = np.sqrt(eval(MSE_updated(y_true, y_pred, return_vec=True)))*48

    validation_MSE = eval(MSE_updated(validation_generator.labels, valid_predictions))
    test_MSE = eval(MSE_updated(test_generator.labels, test_predictions))
    print(f'valid err is {validation_MSE}, test err is {test_MSE}')

    coverages = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.]
    risk_coverages = np.zeros(len(coverages))
    for i, coverage in enumerate(coverages):
        risk_coverages[i] = calc_selective_risk(coverage=coverage, test_losses=test_losses,
                                                uncertainty_on_validation=valid_uncertainty,
                                                uncertainty_on_test=test_uncertainty)
        # print(risk_covera/ges)

    print(f'risk coverage numbers for {my_regressor.name} {uncertainty_metric} is {risk_coverages}')
    [print(str(c)) for c in risk_coverages]
    print(f'AURC-CURVE is {np.sum(risk_coverages)}')


