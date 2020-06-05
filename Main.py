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
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from AutoEncoder import AutoEncoder
from CI import run_ci
from Clustering import Clustering
from Compare import run_compare
from Params import dbscan_param_select
from Evaluation import Evaluation

# Debug tags.
TAG = '[MAIN]'
TAG_RESULT = '[RESULT]'
TAG_ERROR = '[ERROR]'
TAG_WARNING = '[WARNING]'

# Data names.
USPS = 'USPS'
R10K = 'R10K'
MNIST = 'MNIST'
FMNIST = 'FMNIST'
IMDB = 'IMDB'
CIFAR100 = 'CIFAR100'

# Modes.
CLS = 'CLS'
DAE = 'DAE'
DC = 'DC'

# Algorithm names.
KMEANS = 'KMEANS'
GMM = 'GMM'
HIER = 'HIER'
DBSCAN = 'DBSCAN'
ENSEMBLE = 'ENSEMBLE'

# Directory names.
TMP_DIR = 'Weights/'
RES_DIR = 'Results/'


def rho_search(low, high, mem, Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs, lbda, gamma,
               opt,activation, clusters, eps, min_points):

    def eval_rho(rho):
        ae = AutoEncoder(data=data, data_name=data_name, layers=layers, encoding_dim=encoding_dim,
                         batch_size=batch_size, test_data=[],
                         epochs=epochs, rho=rho, lbda=lbda,
                         gamma=gamma, optimizer=opt, activation=activation)
        ae.encoder_decoder()

        clustering = Clustering(ae)

        labels = calculate_labels(DC, Alg, clustering, data, [], clusters, eps, min_points)
        labels = [l if l != -1 else max(labels) + 1 for l in labels]

        ev = Evaluation(data, targets, labels)

        return (ev.Accuracy() + ev.NMI() + ev.Purity() + ev.ARI()) / 4

    def mem_check(x, mem):
        if x in mem.keys():
            return mem[x]
        else:
            tmp_rho = eval_rho(x)
            mem[x] = tmp_rho
            return tmp_rho

    print(TAG, low, high)

    mid = (low + high) / 2
    mid_high = (mid + high) / 2
    mid_low = (mid + low) / 2

    print(TAG, mid_low, mid, mid_high)

    ev_l = mem_check(low, mem)
    ev_m = mem_check(mid, mem)
    ev_h = mem_check(high, mem)

    if ev_l <= ev_m and ev_h <= ev_m or high - low < 5:
        return mid
    elif ev_l > ev_m:
        return rho_search(low, mid, mem, Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs, lbda,
                          gamma, opt, activation, clusters, eps, min_points)
    elif ev_h > ev_m:
        return rho_search(mid, high, mem, Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs, lbda,
                          gamma, opt, activation, clusters, eps, min_points)
    else:
        if ev_h >= ev_l:
            return rho_search(mid, high, mem, Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs,
                              lbda, gamma, opt, activation, clusters, eps, min_points)
        else:
            return rho_search(low, mid, mem, Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs,
                              lbda, gamma, opt, activation, clusters, eps, min_points)


def sample_data(X, Y, percentage):
    idxs = np.random.choice(len(X), int(len(X) * percentage))
    sample = X[idxs]
    s_targets = Y[idxs]

    return sample, s_targets


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


def majority_vote(labels_list):
    """
    Makes a majority vote by calculating the statistical mode of each column. In case of no
    majority the example is labeled as a random label.
    :param labels_list: A 2D array, each row is a list of predictions of each model.
    :return: List of combined predictions.
    """

    tmp = np.array(labels_list)
    labels = []

    for i in range(len(labels_list[0])):

        try:
            labels.append(mode(tmp[:, i]))
        except:
            labels.append(random.choice(tmp[:, i]))

    return labels


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
        print(TAG, TAG_ERROR, 'Dataset not supported.')

    np.load = np_load_old

    return X, Y, opt, activation, clusters


def get_rho(Alg, data, targets, data_name, layers, encoding_dim, batch_size, epochs, lbda, gamma, opt,
            activation, clusters, eps, min_points):
    rho = -1

    if Alg == KMEANS:
        rho = 1  # From source paper.
    elif Alg == GMM:
        rho = 1000  # From source paper.
    elif Alg == HIER:
        rho = rho_search(0, 1000, dict(), Alg, data, targets, data_name, layers, encoding_dim, batch_size,
                         epochs, lbda, gamma, opt, activation, clusters, eps, min_points)
    elif Alg == DBSCAN:
        rho = rho_search(0, 1000, dict(), Alg, data, targets, data_name, layers, encoding_dim, batch_size,
                         epochs, lbda, gamma, opt, activation, clusters, eps, min_points)
    elif Alg == ENSEMBLE:
        rho = rho_search(0, 1000, dict(), Alg, data, targets, data_name, layers, encoding_dim, batch_size,
                         epochs, lbda, gamma, opt, activation, clusters, eps, min_points)
    else:
        print(TAG, TAG_ERROR, 'Algorithm not supported in get_rho().')

    return rho


def calculate_labels(Mode, Alg, clustering, data, test_data, clusters, eps, min_points):
    labels = []

    if Mode == CLS:

        if Alg == KMEANS:
            labels = KMeans(clusters).fit_predict(test_data)
        elif Alg == GMM:
            labels = GaussianMixture(n_components=clusters, covariance_type='spherical').fit_predict(test_data)
        elif Alg == HIER:
            labels = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(
                test_data)
        elif Alg == DBSCAN:
            labels = sklearn.cluster.DBSCAN(algorithm='auto', eps=eps, min_samples=min_points).fit_predict(test_data)
        elif Alg == ENSEMBLE:
            l1 = KMeans(clusters).fit_predict(test_data)
            l2 = GaussianMixture(n_components=clusters, covariance_type='full').fit_predict(test_data)
            l3 = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(
                test_data)
            labels = majority_vote([l1, l2, l3])
        else:
            print(TAG_ERROR, TAG_ERROR, '[' + Mode + '] Algorithm not supported in calculate_labels().')

    elif Mode == DAE:

        if Alg == KMEANS:
            labels = clustering.deep_ae_kmeans(data, test_data, clusters)
        elif Alg == GMM:
            labels = clustering.deep_ae_gmm(data, test_data, clusters)
        elif Alg == HIER:
            labels = clustering.deep_ae_hierarchical(data, test_data, clusters)
        elif Alg == DBSCAN:
            labels = clustering.deep_ae_dbscan(data, test_data, eps, min_points)
        elif Alg == ENSEMBLE:
            labels = clustering.deep_ae_ensemble(data, test_data, clusters)
        else:
            print(TAG_ERROR, TAG_ERROR, '[' + Mode + '] Algorithm not supported in calculate_labels().')

    elif Mode == DC:

        if Alg == KMEANS:
            labels = clustering.deep_cluster_kmeans(data, clusters)
        elif Alg == GMM:
            labels = clustering.deep_cluster_gmm(data, clusters)
        elif Alg == HIER:
            labels = clustering.deep_cluster_hierarchical(data, clusters)
        elif Alg == DBSCAN:
            labels = clustering.deep_cluster_dbscan(data, eps=eps, min_points=min_points, clusters=clusters)
        elif Alg == ENSEMBLE:
            labels = clustering.deep_cluster_ensemble(data, clusters)
        else:
            print(TAG_ERROR, TAG_ERROR, '[' + Mode + '] Algorithm not supported in calculate_labels().')

    else:
        print(TAG, TAG_ERROR, 'Mode not supported in calculate_labels().')

    return labels


def get_test_labels(Alg, ae, test_x, clusters, eps, min_points):

    compressed_data = ae.encoder.predict(test_x)
    labels = []

    if Alg == KMEANS:
        labels = KMeans(clusters).fit_predict(compressed_data)
    elif Alg == GMM:
        labels = GaussianMixture(n_components=clusters, covariance_type='spherical').fit_predict(compressed_data)
    elif Alg == HIER:
        labels = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(
            compressed_data)
    elif Alg == DBSCAN:
        labels = sklearn.cluster.DBSCAN(algorithm='auto', eps=eps, min_samples=min_points).fit_predict(compressed_data)
    elif Alg == ENSEMBLE:
        l1 = KMeans(clusters).fit_predict(compressed_data)
        l2 = GaussianMixture(n_components=clusters, covariance_type='full').fit_predict(compressed_data)
        l3 = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(
            compressed_data)
        labels = majority_vote([l1, l2, l3])
    else:
        print(TAG_ERROR, TAG_ERROR, '[' + Alg + '] Algorithm not supported in get_test_labels().')

    return labels


def main(argv):

    def get_tf(corpus):
        tf_x = np.array([[list(document).count(word) / len(document) for word in set(document)] for document in corpus])

        n = 0.0

        for t in tf_x:
            if len(t) > n:
                n = len(t)

        for i in range(len(tf_x)):
            pad = ([0.0] * (n - len(tf_x[i])))
            tf_x[i] = tf_x[i] + pad

        return np.array(tf_x)

    def majority_classifier(classes, test):
        count_dict = {k:list(classes).count(k) for k in set(classes)}
        return [sorted(count_dict, key=count_dict.get, reverse=True)[0]] * len(test)

    Sample = float(argv[1])
    Data = argv[2]  # USPS, R10K, MNIST, FMNIST, IMDB, CIFAR100
    Mode = argv[3]  # CLS, DAE, DC
    Alg = argv[4]  # KMEANS, GMM, HIER, DBSCAN, ENSEMBLE

    if Sample < 0.0:
        Sample = 0.2
        print(TAG, TAG_WARNING, 'Sample is less then 0, will be set to default value 0.2.')

    if Sample > 1.0:
        Sample = 1.0
        print(TAG, TAG_WARNING, 'Sample is more then 1, will be set to 1.0.')

    print(TAG, 'Main arguments:', Sample, Data, Mode, Alg)

    ########## Load datasets. ##########
    print(TAG, 'Loading data.')

    X, Y, opt, activation, clusters = load_data_and_params(Data)

    whole_X = X
    whole_Y = Y

    X, Y = sample_data(X, Y, Sample)

    (learn_x, test_x) = split_data(X)
    (learn_y, test_y) = split_data(Y)

    eps, min_points = -1, -1

    if Alg == DBSCAN:
        if Data == IMDB:
            xtmp = get_tf(learn_x)
            eps, min_points = dbscan_param_select(xtmp, learn_y)
        else:
            eps, min_points = dbscan_param_select(learn_x, learn_y)

    print(TAG, Data, 'loaded.')
    print(TAG, 'Optimizer:', opt, 'Activation:', activation)
    print(TAG, 'DBSCAN params min_points:', min_points, 'and Epsilon:', eps)
    print(TAG, 'Number of clusters:', clusters)
    ####################################

    save_name = '_' + Data + '_' + Mode + '_' + Alg

    data = learn_x
    test_data = test_x
    targets = test_y

    if Mode == DC:
        targets = learn_y

    # IO layer size.
    io = len(data[0])
    layers = [io, 500, 500, 2000, 10, 2000, 500, 500, io]
    encoding_dim = 10

    # AutoEncoder.
    epochs = 200
    batch_size = 100

    lbda = 0.5
    gamma = 0.5

    rho = 1

    # Sample for rho search ###

    sx, sy = sample_data(learn_x, learn_y, 0.1)

    ###########################

    if Mode == DC:
        rho = get_rho(Alg, sx, sy, Data, layers, encoding_dim, batch_size, 50, lbda, gamma, opt, activation, clusters,
                      eps, min_points)

    print(TAG, 'Rho:', rho)

    ae = AutoEncoder(data=data, data_name=Data, layers=layers, encoding_dim=encoding_dim, batch_size=batch_size,
                     epochs=epochs, rho=rho, lbda=lbda, test_data=test_x,
                     gamma=gamma, optimizer=opt, activation=activation)
    ae.encoder_decoder()

    clustering = Clustering(ae)

    # Learning set.
    acc = 0.0
    nmi = 0.0
    pur = 0.0
    ari = 0.0
    con = 1.0

    # Majority classifier.
    acc_mc = 0.0
    nmi_mc = 0.0
    pur_mc = 0.0
    ari_mc = 0.0

    # Test set.
    acc_t = 0.0
    nmi_t = 0.0
    pur_t = 0.0
    ari_t = 0.0

    # Whole set.
    acc_s = 0.0
    nmi_s = 0.0
    pur_s = 0.0
    ari_s = 0.0

    for i in range(10):
        print(TAG, 'Calculating labels.')

        labels = calculate_labels(Mode, Alg, clustering, data, test_data, clusters, eps, min_points)
        mc_labels = majority_classifier(learn_y, test_y)

        if Alg == DBSCAN or Alg == ENSEMBLE:
            labels = [l if l != -1 else max(labels) + 1 for l in labels]
            mc_labels = [l if l != -1 else max(labels) + 1 for l in mc_labels]

        print(TAG, 'Evaluating learning set labels.')
        ev = Evaluation(data, targets, labels)

        print(TAG, 'Evaluating majority classifier labels.')
        ev_mc = Evaluation(data, mc_labels, test_y)

        # Save weights for NOFM for DC algorithms.
        if Mode == DC:
            weights_path = TMP_DIR + 'weights' + save_name + str(i) + '.h5'
            clustering.ae.model.save_weights(weights_path)

            clusters_tmp = clusters

            if Data == IMDB:
                clusters_tmp = clusters_tmp + 1

            if Data == CIFAR100:
                clusters_tmp = clusters_tmp // 20

            indices_path = run_ci(weights_path, clusters_tmp)
            print('\n' + TAG, 'Indices path:', indices_path)

            con = min(con, run_compare(learn_x, test_x, learn_y, test_y, indices_path, Data, Mode, Alg, io))
            clustering.ae.construct_quality = con

            labels_t = get_test_labels(Alg, ae, test_x, clusters, eps, min_points)
            labels_s = get_test_labels(Alg, ae, whole_X, clusters, eps, min_points)
            
            if Alg == DBSCAN:
                labels_t = [l if l != -1 else max(labels) + 1 for l in labels_t]
                labels_s = [l if l != -1 else max(labels) + 1 for l in labels_s]

            print(TAG, 'Evaluating testing set labels.')
            ev_t = Evaluation(test_x, test_y, labels_t)

            print(TAG, 'Evaluating whole set labels.')
            ev_s = Evaluation(whole_X, whole_Y, labels_s)

            acc_t = max(acc_t, ev_t.Accuracy())
            nmi_t = max(nmi_t, ev_t.NMI())
            pur_t = max(pur_t, ev_t.Purity())
            ari_t = max(ari_t, ev_t.ARI())

            acc_s = max(acc_s, ev_s.Accuracy())
            nmi_s = max(nmi_s, ev_s.NMI())
            pur_s = max(pur_s, ev_s.Purity())
            ari_s = max(ari_s, ev_s.ARI())

        acc = max(acc, ev.Accuracy())
        nmi = max(nmi, ev.NMI())
        pur = max(pur, ev.Purity())
        ari = max(ari, ev.ARI())

        acc_mc = max(acc_mc, accuracy_score(mc_labels, test_y))
        nmi_mc = max(nmi_mc, ev_mc.NMI())
        pur_mc = max(pur_mc, ev_mc.Purity())
        ari_mc = max(ari_mc, ev_mc.ARI())

        print(TAG, '[LEARN] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f, CON: %.4f' % (acc, nmi, pur, ari, con))
        print(TAG, '[TEST] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_t, nmi_t, pur_t, ari_t))
        print(TAG, '[WHOLE] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_s, nmi_s, pur_s, ari_s))
        print(TAG, '[MC] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_mc, nmi_mc, pur_mc, ari_mc))
        print(TAG, 'Progress: ', i + 1, ' / 10')

    print(TAG, TAG_RESULT, '[LEARN] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f, CON: %.4f' % (acc, nmi, pur, ari, con))
    print(TAG, TAG_RESULT, '[TEST] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_t, nmi_t, pur_t, ari_t))
    print(TAG, TAG_RESULT, '[WHOLE] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_s, nmi_s, pur_s, ari_s))
    print(TAG, TAG_RESULT, '[MC] Acc: %.4f, NMI: %.4f, Pur: %.4f, ARI: %.4f' % (acc_mc, nmi_mc, pur_mc, ari_mc))

    f = open(RES_DIR + Data + '_' + Mode + '_' + Alg + '_Results.txt', 'a')
    f.write('LEARN: ' + str((round(acc, 4), round(nmi, 4), round(pur, 4), round(ari, 4), round(con, 4))) + '\n')
    f.write('TEST: ' + str((round(acc_t, 4), round(nmi_t, 4), round(pur_t, 4), round(ari_t, 4))) + '\n')
    f.write('WHOLE: ' + str((round(acc_s, 4), round(nmi_s, 4), round(pur_s, 4), round(ari_s, 4))) + '\n')
    f.write('MC: ' + str((round(acc_mc, 4), round(nmi_mc, 4), round(pur_mc, 4), round(ari_mc, 4))) + '\n')
    f.close()


if __name__ == '__main__':
    main(sys.argv)
