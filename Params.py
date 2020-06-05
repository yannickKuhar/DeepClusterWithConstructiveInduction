import numpy as np
from joblib.numpy_pickle_utils import xrange


def dbscan_param_select(data, targets):

    def get_min_points(classes):
        count_dict = {k:list(classes).count(k) for k in set(classes)}
        return count_dict[sorted(count_dict, key=count_dict.get)[0]]

    def get_epsilon(data):
        tot = 0.0

        for i in xrange(data.shape[0] - 1):
            tot += ((((np.array(data[i + 1]) - np.array(data[i])) ** 2).sum()) ** .5).mean()

        return tot / ((data.shape[0] - 1) * (data.shape[0]) / 2.)

    return get_epsilon(data), get_min_points(targets)
