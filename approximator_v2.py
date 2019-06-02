import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def target_function(x):
    mu = 5
    sigma = 0.3
    return 100*gaussian(x, mu, sigma)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



class Regressor(nn.Module):
    """
    this class will represent the regressor, which will output the
    regression score and his confidence
    """

    def __init__(self, input_size, HL1_sz=256, HL2_sz=128):
        """
        the constructor
        """

        super(Regressor, self).__init__()
        self.input_layer = nn.Linear(input_size, HL1_sz)
        self.bn_after_input = nn.BatchNorm1d(HL1_sz)

        # self.hidden_layer1 = nn.Linear(HL1_sz, HL2_sz)
        # self.hidden_layer2 = nn.Linear(HL2_sz, HL2_sz)
        # self.hidden_layer3 = nn.Linear(HL2_sz, HL2_sz)

        self.regression_layer = nn.Linear(HL2_sz, 1)
        self.confidence_layer = nn.Linear(HL2_sz, 1)
        self.optimizer = None

    def set_optimizer(self, optimizer):
        """
        :param optimizer: optimerzer object from the torch.optim library
        :return: None
        """
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise ValueError(' the given optimizer is not supported'
                             'please provide an optimizer that is an instance of'
                             'torch.optim.Optimizer')
        self.optimizer = optimizer

    def forward(self, input):
        """
        defines the forward pass in the shallow NN we defined
        :param input:
        :return: regression_output, confidence
        """
        x = F.relu(self.input_layer(torch.tensor(input, dtype=torch.float)))
        x = self.bn_after_input(x)
        # x = F.relu(self.hidden_layer1(x))
        # x = F.relu(self.hidden_layer2(x))
        # x = F.relu(self.hidden_layer3(x))

        return self.regression_layer(x), self.confidence_layer(x), x

    def train(self):
        """

        :return:
        """

        x_bat, y_target = generate_batch(256)
        regression = []
        confidence = []
        embeddings = []
        y, c, e = self.forward(x_bat)

        # for i in x_bat:
        #     y, c, e = self.forward(i)
        #     regression.append(y)
        #     confidence.append(c)
        #     embeddings.append(e)

        # embeddings = torch.stack(embeddings)
        # embeddings = embeddings-embeddings.mean()
        loss = F.smooth_l1_loss(y.squeeze(), torch.tensor(y_target, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss



def plot_reg(regressor, target_function, loss):
    """

    :param regressor:
    :param target_function:
    :param loss:
    :return:
    """
    x = np.linspace(0,10, 1000)
    y = []
    y_pred = []
    for i in x:
        y_pred.append(regressor(i)[0])
        y.append(target_function(i))

    plt.plot(x, y)
    plt.plot(x, y_pred)
    plt.legend(['true function', 'approximator'])
    plt.title(f'loss is {loss}')


def split_data(data: pd.DataFrame, train_percent, valid_percent):
    """

    :param data:
    :param train_percent:
    :param valid_percent:
    :return:
    """

    train_data_len = int(data.shape[0]*train_percent)
    valid_data_len = int(data.shape[0]*valid_percent)
    test_data_len = data.shape[0] - train_data_len - valid_data_len

    num_lines_used = train_data_len
    train_data = np.array(data.iloc[:num_lines_used, :-1]),\
                 np.array(data.iloc[:num_lines_used, -1])
    valid_data = np.array(data.iloc[num_lines_used: valid_data_len + num_lines_used, :-1]),\
                 np.array(data.iloc[num_lines_used: valid_data_len + num_lines_used, -1])
    num_lines_used += valid_data_len
    test_data = np.array(data.iloc[num_lines_used: test_data_len + num_lines_used, :-1]),\
                 np.array(data.iloc[num_lines_used: test_data_len + num_lines_used, -1])

    return train_data, valid_data, test_data


data = pd.read_excel('./Concrete_data.xls')
train_data, valid_data, test_data = split_data(data, 0.5, 0.25)
np.random.seed(0)


def generate_batch(batch_size):

    global train_data
    train_data_x, train_data_y = train_data
    rand_perm = np.random.permutation(train_data_y.shape[0])
    rand_perm = rand_perm[:batch_size]
    return train_data_x[rand_perm, :], train_data_y[rand_perm]


def calc_validation_loss(regressor, validation_data):

    validation_data_x, validation_data_y = validation_data
    y_pred,_ ,_  = regressor(validation_data_x)

    return F.smooth_l1_loss(y_pred.squeeze(), torch.tensor(validation_data_y, dtype=torch.float32))


if __name__ == '__main__':

    r = Regressor(8, 64, 64)
    optimizer = optim.Adam(r.parameters(), lr=3*1e-2)
    r.set_optimizer(optimizer)

    data = pd.read_excel('./Concrete_data.xls')

    num_batches = 1000
    for i in range(num_batches):
        loss = r.train()
        valid_loss = calc_validation_loss(r, valid_data)
        print(f'training_loss: {loss.item()}  validation_loss: {valid_loss}')


