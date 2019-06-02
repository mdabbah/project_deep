import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    mu = 5
    sigma = 0.3
    return 100*gaussian(x, mu, sigma)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

batch_size = 150
x_sampes = np.append(np.linspace(0,10, 10) ,0.3*np.random.randn(70)+5)
# x_sampes =0.3*np.random.randn(70)+5
y_samples = target_function(x_sampes)
def generate_batch(batch_size):
    return x_sampes, y_samples


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
        x = F.relu(self.input_layer(torch.tensor([input])))
        # x = F.relu(self.hidden_layer1(x))
        # x = F.relu(self.hidden_layer2(x))
        # x = F.relu(self.hidden_layer3(x))

        return self.regression_layer(x), self.confidence_layer(x), x

    def train(self):
        """

        :return:
        """

        x_bat, y_target = generate_batch(32)
        regression = []
        confidence = []
        embeddings = []
        for i in x_bat:
            y, c, e = self.forward(i)
            regression.append(y)
            confidence.append(c)
            embeddings.append(e)

        # embeddings = torch.stack(embeddings)
        # embeddings = embeddings-embeddings.mean()
        loss = F.smooth_l1_loss(torch.stack(regression).flatten(), torch.tensor(y_target, dtype=torch.float32))
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


if __name__ == '__main__':

    r = Regressor(1, 64, 64)
    optimizer = optim.Adam(r.parameters(), lr=3*1e-2)
    r.set_optimizer(optimizer)

    num_batches = 1000

    fig = plt.figure()
    for i in range(num_batches):
        loss = r.train()
        plot_reg(r, target_function, loss)
        print(loss)
        fig.clf()

    plt.show()