from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as io
import random
import time

class DataPrep:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.y_list = None
        self.y_test_list = None

        self.input_shape = None
        self.output_size = None

    def cifar10(self, train_size=None, test_size=None):
        self.input_shape = [32, 32, 3]
        self.output_size = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def cifar100(self, train_size=None, test_size=None):
        self.input_shape = [32, 32, 3]
        self.output_size = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def mnist(self, train_size=None, test_size=None):
        self.input_shape = [28, 28, 1]
        self.output_size = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def add_padding(self, border_width=2):

        pad = np.zeros([self.x.shape[0],
                        self.x.shape[1] + border_width*2,
                        self.x.shape[2] + border_width*2,
                        self.x.shape[3]])
        pad[:, border_width:-border_width, border_width:-border_width, :] = self.x

        pad_t = np.zeros([self.x_test.shape[0],
                          self.x_test.shape[1] + border_width * 2,
                          self.x_test.shape[2] + border_width * 2,
                          self.x_test.shape[3]])
        pad_t[:, border_width:-border_width, border_width:-border_width, :] = self.x_test

        self.x = pad
        self.x_test = pad_t

    def grayscale2rgb(self):
        x = np.array([np.squeeze(self.x), np.squeeze(self.x), np.squeeze(self.x)])
        x_test = np.array([np.squeeze(self.x_test), np.squeeze(self.x_test), np.squeeze(self.x_test)])

        x = np.moveaxis(x, 0, -1)
        x_test = np.moveaxis(x_test, 0, -1)

        self.x = x
        self.x_test = x_test

    def fashion_mnist(self, train_size=None, test_size=None):
        self.input_shape = [28, 28, 1]
        self.output_size = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def cSVHN_ez(self, validation_split=0.3):
        self.input_shape = [32, 32, 3]
        self.output_size = 10

        data_dict = io.loadmat('../datasets/cSVHN/extra_32x32.mat')
        data_dict['X'] = np.moveaxis(data_dict['X'], -1, 0)
        data_dict['y'][data_dict['y'] == 10] = 0
        data_dict['y'] = to_categorical(data_dict['y'], 10)

        train_size = int(round(data_dict['X'].shape[0]*(1-validation_split)))

        self.x = data_dict['X'][:train_size]
        self.x_test = data_dict['X'][train_size:]

        self.y = data_dict['y'][:train_size]
        self.y_test = data_dict['y'][train_size:]


    def cSVHN(self, validation_split=0.3):
        self.input_shape = [32, 32, 3]
        self.output_size = 10

        data_dict = io.loadmat('../datasets/cSVHN/train_32x32.mat')
        data_dict['X'] = np.moveaxis(data_dict['X'], -1, 0)
        data_dict['y'][data_dict['y'] == 10] = 0
        data_dict['y'] = to_categorical(data_dict['y'], 10)

        train_size = int(round(data_dict['X'].shape[0]*(1-validation_split)))

        self.x = data_dict['X'][:train_size]
        self.x_test = data_dict['X'][train_size:]

        self.y = data_dict['y'][:train_size]
        self.y_test = data_dict['y'][train_size:]

    def get_class(self, label):
        if label > self.y.shape[1] or label < 0 : return None

        ind   = np.ix_(self.y[:, label] == 1)
        ind_t = np.ix_(self.y_test[:, label] == 1)

        return self.x[ind], self.y[ind], self.x_test[ind_t], self.y_test[ind_t]

    def shuffle_data(self, x, y, x_test, y_test):
        x = np.concatenate(x)
        y = np.concatenate(y)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        training = list(zip(x, y))
        validation = list(zip(x_test, y_test))

        random.shuffle(training)
        random.shuffle(validation)

        x, y = zip(*training)
        x_test, y_test = zip(*validation)

        return np.array(x), np.array(y), np.array(x_test), np.array(y_test)

    def sample_dataset(self, labels):
        x = []
        y = []
        x_test = []
        y_test = []

        for l in labels:
            xi, yi, x_testi, y_testi = self.get_class(l)
            x.append(xi)
            y.append(yi)
            x_test.append(x_testi)
            y_test.append(y_testi)
        x, y, x_test, y_test = self.shuffle_data(x, y, x_test, y_test)

        y = y[:, ~np.all(y == 0, axis=0)]
        y_test = y_test[:, ~np.all(y_test == 0, axis=0)]

        return x, y, x_test, y_test

    def add_noise(self, noise_factor=0.5, prob=0.35):

        N, wi, he, ch = self.x.shape
        noise  = np.random.rand(N, wi, he, ch)
        self.x[noise < prob] = 0                    # Pepper
        self.x[noise < prob*noise_factor] = 255     # Salt

        N, wi, he, ch = self.x_test.shape
        noise = np.random.rand(N, wi, he, ch)
        self.x_test[noise < prob] = 0                    # Pepper
        self.x_test[noise < prob*noise_factor] = 255     # Salt

    def normalize(self):
        self.x = np.divide(self.x, 255)
        self.x_test = np.divide(self.x_test, 255)


if __name__ == "__main__":
    mn = DataPrep()
    mn.mnist()
    mn.padded_mnist()
    mn.grayscale2rgb()

    svhn = DataPrep()
    svhn.cSVHN_ez()


    plt.figure('svhn normalized: Y=' + str(svhn.y[0]))
    plt.imshow(svhn.x[0]/255)
    plt.show(block=False)

    plt.figure('mnist: Y='+str(mn.y[0]))
    plt.imshow(mn.x[0]/255)
    plt.show(block=True)
