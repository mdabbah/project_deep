import os

from keras import Model, Input
import numpy as np
from keras.layers import Dropout, Lambda, Dense

from evaluate_distance_based import pickle_embeddings, unpickle_embeddings
from train_distance_based_regression import MSE_updated


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

    return np.mean(test_losses[uncertainty_on_test < th]) / np.mean(uncertainty_on_test < th)


def turn_on_dropout(model: Model):
    """
    since keras doesn't have the option to turn on dropout on testing,
    my workaround is to replace the dropout layers which lambda layers that use dropout function
    that way it is always turned on
    :param model: model to replace the dropout layers in
    :return: the updated model
    """
    layers = [l for l in model.layers]

    x = input =Input(shape=model.input_shape[1:])
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            print(layer.rate)
            x = Dropout(rate=layer.rate, name=layer.name + '_always_on')(x, training=True)
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

    predictions = np.zeros((num_evaluations, *test_generator.key_points.shape))
    model_dropout_turned_on = turn_on_dropout(my_regressor)
    for i in range(num_evaluations):
        predictions[i, :] = model_dropout_turned_on.predict_generator(test_generator)

    return np.mean(predictions, axis=0), np.mean(np.std(predictions, axis=0), axis=-1)


def distance_predictions(my_regressor, test_generator):

    training_generator = data_genetator.MYGenerator('train', use_nans=True)
    dataset_name = 'regression/' + test_generator.dataset_name
    model_name = my_regressor.name
    pkl = unpickle_embeddings(model_name, dataset_name)
    if not pkl:
        pickle_embeddings(my_regressor, model_name, dataset_name, training_generator)
        pkl = unpickle_embeddings(model_name, dataset_name)

    training_embeddings, training_targets = pkl
    layer_name = 'embedding'
    encoder_model = Model(inputs=my_regressor.input,
                          outputs=my_regressor.get_layer(layer_name).output)
    test_embeddings = encoder_model.predict_generator(test_generator)

    batch_size = 32
    num_training_samples = training_embeddings.shape[0]
    num_test_samples = test_embeddings.shape[0]
    embedding_len_per_target = int(training_embeddings.shape[1]//training_targets.shape[1])  # we have y_true.shape[0] targets

    dists = np.zeros((test_embeddings.shape[0], training_embeddings.shape[0], training_targets.shape[1]))
    mask = np.repeat(np.expand_dims(np.isnan(training_targets), axis=0), batch_size, axis=0)
    for bt in range(np.int(np.ceil(num_test_samples//batch_size))):
        test_embeddings_bt = test_embeddings[bt*batch_size:batch_size*(bt+1), :]

        batch_dists = np.sum(
            np.reshape(np.square(np.expand_dims(training_embeddings, 0) - np.expand_dims(test_embeddings_bt, 1)),
                       newshape=(batch_size, num_training_samples, -1, embedding_len_per_target)), axis=-1)
        batch_dists[mask] = np.inf
        dists[batch_size*bt:batch_size*(bt+1), :, :] = batch_dists

    return my_regressor.predict_generator(test_generator), np.mean(np.min(dists, axis=1), axis=-1)


if __name__ == '__main__':

    # general params
    data_set = 'facial_key_points'
    loss_type = 'MSE'  # options 'l1_smoothed', 'distance_classifier'
    arch = 'ELU_arch'
    uncertainty_metric = 'min_distance'  # options 'min_distance' , 'MC_dropout_std'
    batch_size = 32
    input_size = None
    loss_function = MSE_updated
    my_regressor = None

    # loading data
    if data_set == 'facial_key_points':
        import facial_keypoints_data_generator as data_genetator  # choose data set
        input_size = 96, 96, 3
        print("training on facial keypoints")
    elif data_set == 'fingers':
        raise ValueError("not supported yet")

    # building the model
    if data_set.startswith('facial'):
        from distance_classifier import DistanceClassifier
        base_model = DistanceClassifier(input_size, num_classes=None, include_top=False)
        x = base_model.output
        x = Dense(30, activation='linear')(x)
        my_regressor = Model(base_model.input, x, name=f'{data_set} regression model')
    optimizer = 'adadelta'

    my_regressor.compile(optimizer=optimizer, loss=loss_function, metrics=['MSE'])

    # load weights
    weights_path = r'.\results\regression\MSEs_ELU_arch_facial_key_points\MSE_ELU_arch_ 27_0.042_0.042_0.00701_ 0.00767.h5'
    my_regressor.name = os.path.split(weights_path)[-1][:-3]
    my_regressor.load_weights(weights_path)

    validation_generator = data_genetator.MYGenerator('valid', batch_size, shuffle=True, use_nans=True)
    test_generator = data_genetator.MYGenerator('test', batch_size, shuffle=True, use_nans=True, horizontal_flip_prob=0)

    if uncertainty_metric == 'MC_dropout_std':
        valid_predictions, valid_uncertainty = MC_dropout_predictions(my_regressor, validation_generator)
        test_predictions, test_uncertainty = MC_dropout_predictions(my_regressor, test_generator)
    else:
        valid_predictions, valid_uncertainty = distance_predictions(my_regressor, validation_generator)
        test_predictions, test_uncertainty = distance_predictions(my_regressor, test_generator)

    test_losses = MSE_updated(test_generator.key_points, test_predictions, return_vec=True)

    coverages = [0.7, 0.8, 0.85, 0.9, 0.95]
    risk_coverages = np.zeros(len(coverages))
    for i, coverage in enumerate(coverages):
        risk_coverages[i] = calc_selective_risk(coverage=0.7, test_losses=test_losses,
                                                uncertainty_on_validation=valid_uncertainty,
                                                uncertainty_on_test=test_uncertainty)

    print(f'AURC-CURVE is {np.sum(risk_coverages)}')


