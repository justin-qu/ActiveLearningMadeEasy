from tensorflow.keras import datasets
import numpy as np

__all__ = ['Dataset', 'MNIST', 'Fashion_MNIST']

class Dataset:
    def __init__(self, name, train_data, train_labels, test_data, test_labels):
        self.name           = name
        self.train_data     = train_data
        self.train_labels   = train_labels
        self.test_data      = test_data
        self.test_labels    = test_labels

    def __str__(self):
        return '{} Dataset'.format(self.name)

    ## Shuffle Training Data
    def shuffle(self):
        p = np.random.permutation(len(self.train_data))
        self.train_data     = self.train_data[p]
        self.train_labels   = self.train_labels[p]



## MNIST Dataset
class MNIST(Dataset):
    def __init__(self, normalize=False):
        (train_data, train_labels), (test_data, test_labels) = datasets.mnist.load_data()

        if normalize:
            train_data = train_data.reshape(-1, 28, 28, 1) / 255
            test_data  = test_data.reshape(-1, 28, 28, 1) / 255

        super().__init__('MNIST', train_data, np.array(train_labels), test_data, np.array(test_labels))



## Fashion MNIST Dataset
class Fashion_MNIST(Dataset):
    def __init__(self, normalize=False):
        (train_data, train_labels), (test_data, test_labels) = datasets.fashion_mnist.load_data()

        if normalize:
            train_data = train_data.reshape(-1, 28, 28, 1) / 255
            test_data  = test_data.reshape(-1, 28, 28, 1) / 255

        super().__init__('Fashion_MNIST', train_data, np.array(train_labels), test_data, np.array(test_labels))
