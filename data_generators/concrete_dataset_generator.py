import keras
import pandas as pd
import numpy as np


class MYGenerator(keras.utils.Sequence):

    DATAPATH=r'.\data\regression\concrete_data'

    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False):

        self.dataset_name = 'concrete_strength'
        self.data_path = f'{self.DATAPATH}/{data_type}.csv'
        self.data = pd.read_csv(self.data_path)

        if shuffle:
            self.data = self.data.sample(frac=1)

        self.x_data = np.array(self.data.iloc[:, :-1])
        self.y_data = self.gt = np.array(self.data.iloc[:, -1])
        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.y_data) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.x_data[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        batch_y = self.y_data[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_x, batch_y

    def on_epoch_end(self):
        print('shuffled on epoch end')
        prem = np.random.permutation(self.x_data.shape[0])
        self.x_data = self.x_data[prem]
        self.y_data = self.y_data[prem]
        self.gt = self.y_data
