from keras import Model
import numpy as np
from keras.layers import Dropout, Lambda, Dense
from train_distance_based_regression import  MSE_updated


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

    x = layers[0].output
    for layer in model.layers:
        if layer.name.startswith('dropout'):
            print(layer.rate)
            x = Dropout(rate=layer.rate, name=layer.name + 'always_on')(x, training=True)
            continue
        x = layer(x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model


if __name__=='__main_-':

    # general params
    data_set = 'facial_key_points'
    loss_type = 'MSE'  # options 'l1_smoothed', 'distance_classifier'
    arch = 'ELU_arch'
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
    weights_path = r'.\results\regression\MSEs_ELU_arch_facial_key_points\MSE_ELU_arch_ 591_0.004_0.004_0.00447_ 0.00400.h5'
    my_regressor.load_weights(weights_path)

    validation_generator = data_genetator.MYGenerator('valid', batch_size, shuffle=True)
    test_generator = data_genetator.MYGenerator('test', batch_size, shuffle=True)


