import sys
import math
import random
import sklearn
import numpy as np

from statistics import mode
from keras.datasets import cifar100, fashion_mnist, imdb
from keras.preprocessing import sequence
from mat4py import loadmat
from skimage import color
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from Evaluation import Evaluation
from sklearn.metrics import accuracy_score


# Data names.
USPS = 'USPS'
R10K = 'R10K'
MNIST = 'MNIST'
FMNIST = 'FMNIST'
IMDB = 'IMDB'
CIFAR100 = 'CIFAR100'


def load_data_and_params(Data):
    X = []
    Y = []

    eps = -1.0
    min_points = -1

    opt = 'adadelta'
    activation = 'sigmoid'

    np_load_old = np.load
    clusters = 10

    if Data == USPS:
        usps = loadmat('Data/uspsdata.mat')
        X = np.array(usps['clusterdata'])
        Y = np.array(usps['clustertargets'])
        Y = np.array([int(np.where(t == 1)[0]) for t in Y])
    elif Data == R10K:
        reuters10k = loadmat('Data/reutersidx10kdata.mat')
        X = np.array(reuters10k['clusterdata'])
        Y = np.array(reuters10k['clustertargets'])
        Y = np.array([int(np.where(t == 1)[0]) for t in Y])
        opt = 'sgd'
        activation = 'relu'
        clusters = 4
    elif Data == FMNIST:
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        X = np.concatenate((train_images, test_images))
        Y = np.concatenate((train_labels, test_labels))
        X = np.array([f.flatten() for f in X]) / 255
    elif Data == MNIST:
        mnist = loadmat('Data/mnist_fulldata.mat')
        X = np.array(mnist['clusterdata'])
        Y = np.array(mnist['clustertargets'])
        Y = np.array([int(np.where(t == 1)[0]) for t in Y])
    elif Data == IMDB:
        # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
        max_words = 500
        X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')
        X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        clusters = 2
    elif Data == CIFAR100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_train = np.array([color.rgb2gray(x).flatten() for x in x_train])
        x_test = np.array([color.rgb2gray(x).flatten() for x in x_test])
        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))
        Y = Y.flatten()
        clusters = 100
    else:
        print('[MC] Dataset not supported.')

    np.load = np_load_old

    return X, Y, opt, activation, clusters


def split_data(dataset, margin=0.7):
    """
    Splits the dataset into learning and testing subsets.
    :param margin: Determines how the dataset will be split
        i.e. how many elements will be in the learning set. By default,
        the parameter is set to 0.7 to make the standard 70:30 split.
    :param dataset: The data we ant to split.
    :return (learn, test): Learning and testing set.
    """

    n = math.ceil(len(dataset) * margin)

    learn = dataset[:n]
    test = dataset[n:]

    return learn, test


def main(argv):

    def majority_classifier(classes, test):
        count_dict = {k:list(classes).count(k) for k in set(classes)}
        return [sorted(count_dict, key=count_dict.get, reverse=True)[0]] * len(test)

    Data = argv[1]  # USPS, R10K, MNIST, FMNIST, IMDB, CIFAR100

    print('[MC] Main arguments:', Data)

    ########## Load datasets. ##########
    print('[MC] Loading data.')

    X, Y, opt, activation, clusters = load_data_and_params(Data)

    (learn_x, test_x) = split_data(X)
    (learn_y, test_y) = split_data(Y)

    print(set(test_y))

    print('[MC]', Data, 'loaded.')
    print('[MC] Number of clusters:', clusters)
    ####################################

    mc_labels = majority_classifier(learn_y, test_y)

    print(mc_labels)
    ev_mc = Evaluation(Data, mc_labels, test_y)

    acc_mc = accuracy_score(mc_labels, test_y)
    nmi_mc = ev_mc.NMI()
    pur_mc = ev_mc.Purity()
    ari_mc = ev_mc.ARI()

    print('[MC] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_mc, nmi_mc, pur_mc, ari_mc))


if __name__ == '__main__':
    main(sys.argv)
