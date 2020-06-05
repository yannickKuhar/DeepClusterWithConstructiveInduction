import sys
import h5py
import sklearn
import math
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans

TAG = '[CI]'

TMP_RULES = 'Rules/'
TMP_IDX = 'Indices/'


def cluster_weights(weights, n_clusters):
    best_labels = []
    best_eval = 0.0
    n = 0

    X = weights.reshape(-1, 1)

    for i in range(2, n_clusters):

        labels = KMeans(i).fit_predict(X)
        score = metrics.silhouette_score(X, labels, metric='euclidean')

        if score > best_eval:
            best_eval = score
            best_labels = labels
            n = i

    groups = np.array([weights[np.where(best_labels == i)] for i in range(n)])
    indices = np.array([np.where(best_labels == i)[0] for i in range(n)])

    # print(TAG, indices)
    # print(TAG, groups)
    return indices, groups


def get_rule_and_write(avgs, bias, indices, file_name, indices_file_name):
    n = len(avgs)

    rule = 'if '

    used_avgs = []
    usable_idxs = []

    for i in range(n):

        if avgs[i] * len(indices[i]) > bias:

            usable_idxs.append(i)

            if i > 0 and rule != 'if ':
                rule = rule + ' + '

            rule = rule + str(avgs[i]) + ' * NumberTrue(' + ', '.join(map(str, indices[i])) + ')'
            used_avgs.append(i)

    if rule == 'if ':
        rule = 'No rule.\n'
    else:
        rule = rule + ' > ' + str(bias) + ' then activate neuron\n'

    if len(used_avgs) == 1 and (math.ceil(abs(bias) / abs(avgs[used_avgs[0]]))) < len(indices[used_avgs[0]]):
        rule = 'if ' + str(math.ceil(abs(bias) / abs(avgs[used_avgs[0]]))) + ' of ' + ','.join(
            map(str, indices[used_avgs[0]])) + ') then activate neuron\n'
    else:
        rule = 'No rule.\n'

    f = open(TMP_RULES + file_name, "a")
    g = open(TMP_IDX + indices_file_name, "a")

    idxs = ''

    for i in usable_idxs:
        idxs = idxs + ','.join(map(str, indices[i]))

    f.write(rule)

    if len(usable_idxs) == 0:
        g.write(','.join(map(str, indices[n - 1])))
    else:
        g.write(idxs + '\n')

    f.close()
    g.close()


def run_ci(weights_file, n_clusters):
    f = h5py.File(weights_file, 'r')

    weights_file = weights_file.split('/')[1]
    print(TAG, 'Weights file: ', weights_file)

    result_file_name = weights_file.split('.')[0] + '_rules.txt'
    indices_file_name = weights_file.split('.')[0] + '_indices.csv'

    keys = list(f.keys())[1:]
    # print(TAG, 'Keys:', keys)

    print(TAG, 'Constructive induction start.')

    for k in keys:
        sub_keys = list(f[k])

        for s in sub_keys:
            biases = np.array(list(f[k][s]['bias:0']))
            weights = np.array(list(f[k][s]['kernel:0']))

            for i in range(len(biases)):
                sys.stdout.write('\r' + TAG + ' Progress : ' + str(i + 1) + ' / ' + str(len(biases)))
                sys.stdout.flush()
                # print(TAG, '( ' + k + ' ' + s + ' )' ,i + 1, '/', len(biases), flush=True)
                indices, groups = cluster_weights(weights[:, i], n_clusters)
                avgs = np.array([np.mean(g) for g in groups])
                get_rule_and_write(avgs, biases[i], indices, result_file_name, indices_file_name)

            break
        break

        stdout.write('\n')
        print(TAG, 'Constructive induction done.')

    return TMP_IDX + indices_file_name
