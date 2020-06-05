import csv
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.svm import SVC

TAG = '[COMPARE]'
TAG_RESULT = '[RESULT]'
TAG_ERROR = '[ERROR]'

# Construct dir.
CON_DIR = 'Constructs/'
LAB_DIR = 'Labels/'


def save_labels(path, labels):
    print(TAG, 'Writing result labels to: ' + path)

    with open(path + '.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(labels)

    print(TAG, 'Done writing labels.')


def cluster_data(data, n_clusters):
    best_labels = []
    best_eval = 0.0
    n = 0

    for i in range(2, n_clusters):

        print(TAG, 'Clustering', i, '...')

        labels = KMeans(i).fit_predict(data)
        score = metrics.silhouette_score(data, labels, metric='euclidean')

        if score > best_eval:
            best_eval = score
            best_labels = labels
            n = i

    groups = np.array([data[np.where(best_labels == i)] for i in range(n)])
    indices = np.array([np.where(best_labels == i)[0] for i in range(n)])

    return n, indices, groups


def pad_data(data, element=0.0):
    n = max(len(l) for l in data)

    for i in range(len(data)):
        data[i] = data[i] + [element] * (n - len(data[i]))


def read_indices(file_name):
    indices = []

    f = open(file_name, "r")

    for line in csv.reader(f):
        indices.append((list(map(float, line))))

    f.close()

    return indices


def conf_data(data, conf=0.5):
    construct = []

    for i in range(len(data)):

        candidate = sum(d.count(i) for d in data) / len(data)

        # print(TAG, 'Candidate:', candidate)

        if candidate >= conf:
            construct.append(i)

    return construct


def compare(result, learn_x, test_x, learn_y, test_y, data_name, mode, alg):
    svm_kon = SVC(gamma='auto', verbose=False)

    print(TAG, 'Getting result data.')
    xr_train = np.array([x[result] for x in learn_x])
    xr_test = np.array([x[result] for x in test_x])
    print(TAG, 'Result data done.')

    print(TAG, 'Fitting R -', data_name)
    svm_kon.fit(xr_train, learn_y)
    labels = svm_kon.predict(xr_test)

    con_acc = accuracy_score(labels, test_y)

    save_labels(LAB_DIR + data_name + '_' + mode + '_' + alg + '_svm_labels', labels)

    print(TAG, 'ACC R - ' + data_name, con_acc);

    return 1.0 - con_acc


def write_file(result, file_name, c):
    print(TAG, 'Writing results to .csv file.')

    with open(CON_DIR + file_name + '_construct_' + str(c) + '.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(result)

    print(TAG, 'Done writing:', CON_DIR + file_name + '_construct_' + str(c) + '.csv')


def run_compare(learn_x, test_x, learn_y, test_y, indices_path, Data, Mode, Alg, data_dim):
    data = read_indices(indices_path)

    file_name = indices_path.split('/')[1]

    construct_name = file_name.split('.')[0]

    print(TAG, 'Construct name:', construct_name)

    result = conf_data(data)
    print(TAG, 'Construct len:', len(result))

    write_file(result, construct_name, 0.5)

    return compare(result, learn_x, test_x, learn_y, test_y, Data, Mode, Alg)
