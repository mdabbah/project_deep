import pandas as pd
import numpy as np
import keras


TRAIN_FILE_PATH_WITH_NANS = './data/regression/facial_keypoints/train_with_nans.csv'
TRAIN_FILE_PATH_NO_NANS = './data/regression/facial_keypoints/train_no_nans.csv'

VALID_FILE_PATH_WITH_NANS = './data/regression/facial_keypoints/valid_with_nans.csv'
VALID_FILE_PATH_NO_NANS = './data/regression/facial_keypoints/valid_no_nans.csv'

TEST_FILE_PATH = './data/regression/facial_keypoints/test.csv'


class MYGenerator(keras.utils.Sequence):

    def __init__(self, data_type: str, batch_size: int = 100, shuffle: bool = False,
                 use_nans=False, horizontal_flip_prob=0.5, jitterx_limits=(), jittery_limits=()):

        if data_type == 'train':
            data = pd.read_csv(TRAIN_FILE_PATH_WITH_NANS) if use_nans else pd.read_csv(TRAIN_FILE_PATH_NO_NANS)
        elif data_type == 'valid':
            data = pd.read_csv(VALID_FILE_PATH_WITH_NANS) if use_nans else pd.read_csv(VALID_FILE_PATH_NO_NANS)
        else:
            data = pd.read_csv(TEST_FILE_PATH)

        self.image_dims = 96, 96, 3

        self.images = data['Image']
        # images between 0-1  array of size num_samples X 96*96 X 3
        self.images = np.array([np.fromstring(i, sep=' ') for i in self.images])/255.
        self.images = self.images.reshape([-1, 1, *self.image_dims[:2] ]).repeat(repeats=3, axis=1).transpose([0, 2, 3, 1])

        # those are the 'labels' for regression .. we shift them to be between [-1,1]
        self.key_points = (np.array(data.drop(columns=['Image'])) - self.image_dims[0]/2)/(self.image_dims[0]/2)

        #  this is for the stupidly complicated flipping --
        #  idea is that the right eye is now the left one and vice versa
        # we as humans don't recognize if a human face is horizontally flipped, and that's why
        # some keypoints need to swap places like
        # left_eye_center_x -> right_eye_center_x
        # left_eye_center_y -> right_eye_center_y
        self.flip_indices = [
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
        ]

        if shuffle:
            num_samples = self.images.shape[0]
            permute = np.random.permutation(num_samples)
            self.images = self.images[permute, :, :, :]
            self.key_points = self.key_points[permute, :]

        self.batch_size = batch_size
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self):
        return np.int(np.ceil(len(self.key_points) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.key_points[idx * self.batch_size: (idx + 1) * self.batch_size, :]

        num_samples_to_flip = int(batch_x.shape[0]*self.horizontal_flip_prob)
        if num_samples_to_flip > 0:
            samples_to_flip = np.random.choice(batch_x.shape[0], num_samples_to_flip)

            for a, b in self.flip_indices:
                batch_y[samples_to_flip, a], batch_y[samples_to_flip, b] = (
                    batch_y[samples_to_flip, b], batch_y[samples_to_flip, a])

            batch_y[samples_to_flip, ::2] = batch_y[samples_to_flip, ::2]*-1  # times -1 to flip, since y is centered[-1,1]

            batch_x[samples_to_flip, :, :, :] = batch_x[samples_to_flip, :, ::-1, :]

        return batch_x, batch_y
