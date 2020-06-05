import random 
from statistics import mode

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

TAG = '[CLUSTERING]'


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


def update_Y(Y, U, compressed_data, rho, lbda, gamma, construct_quality, cluster_centers, labels):
    """
    Updates the Y matrix with the formula from the source paper.
    :param Y: The Y matrix a copy of hidden features.
    :param U: The U matrix defined by U = U + Y - compressed_data.
    :param compressed_data: The hidden representation of the input data.
    :param rho: The penalty parameter (rho > 0) which control how close Y and the hidden features are.
    :param lbda: The lambda parameter which defines the trade-off between the network objective and the clustering objective.
    :param gamma: The gamma parameter which defines the constructive inductions influence.
    :param construct_quality: The quality of our construct.
    :param cluster_centers: A list of cluster points which represent the characteristic of each cluster (center or mean point).
    :param labels: Cluster labels.
    :return: Updated Y.
    """
    
    basis = np.array([np.mean(cluster_centers, 0) if l == -1 else cluster_centers[l] for l in labels])
    # basis = np.array([np.ones(cluster_centers, 0) if l == -1 else cluster_centers[l] for l in labels])
    # basis = np.array([np.zeros(len(cluster_centers[0])) if l == -1 else cluster_centers[l] for l in labels])

    return lbda * gamma * basis + rho * (compressed_data - U) / (lbda * gamma * construct_quality + rho)


def get_cluster_means(data, labels):
    """
    Computes the mean point of each cluster.
    :param data: Data to be clustered.
    :param labels: Cluster labels.
    :return: List of mean points of every cluster.
    """
    
    if len(data) != len(labels):
        print(TAG, 'Error in get_cluster_mena()')
        return -1

    n = len(set(labels)) - 1 if -1 in set(labels) else len(set(labels))

    means = np.array([np.zeros(len(data[0]))] * n)
    counts = [0] * n

    for i in range(n):

        if labels[i] != -1:
            means[i] += data[i]
            counts[i] += 1

    for i in range(n):
        if counts[i] > 0:
            means[i] /= counts[i]

    return means


def gaussianND(X, mu, sigma):
    """
    Computes the normal distribution of the dataset.
    :param X: Data to be clustered.
    :param mu: Mean of a cluster.
    :param sigma: Covariance matrix of a cluster. 
    :return: Normal distribution computed from the parameters.
    """
    n = len(X[0])
    reg_lambda = 0.00001
    hdim = len(mu)

    mean_diff = X - mu

    return 1 / np.sqrt((2 * np.pi) ** n * np.linalg.det(sigma)) * np.exp(
        -1 / 2 * np.sum((np.matmul(mean_diff, np.linalg.inv(sigma + reg_lambda * np.eye(hdim))) * mean_diff), 1))


def update_Y_GMM(Y, X, U, means, rho, lbda, gamma_ci, construct_quality, sigmas):
    """
    Updates the Y matrix with the formula from the source paper.
    :param Y: The Y matrix a copy of hidden features.
    :param U: The U matrix defined by U = U + Y - compressed_data.
    :param means: The means of all clusters.
    :param rho: The penalty parameter (rho > 0) which control how close Y and the hidden features are.
    :param lbda: The lambda parameter which defines the trade-off between the network objective and the clustering objective.
    :param gamma_ci: The gamma parameter which defines the constructive inductions influence.
    :param sigmas: The covariance matrices of all clusters.
    """

    Px = np.array([gaussianND(X, means[i], sigmas[i]) for i in range(len(means))])
    tmp = Px * np.pi
    
    sum_of_tmp = np.array([1 if i == 0 else i for i in np.sum(tmp, 1)])

    gamma = tmp.T / sum_of_tmp

    hdim = len(means[0])
    N = len(X)

    num_cluster = len(means)

    inv_sigma = np.zeros(sigmas.shape)

    # print(sigmas.shape)

    for i in range(len(means)):
        inv_sigma[:][:][i] = np.linalg.inv(sigmas[:][:][i] + lbda * gamma_ci * construct_quality * np.eye(hdim))

    for i in range(N):
        left_coeff_matrix = np.zeros((hdim, hdim))
        right_side = rho * (X[i] - U[i])

        for k in range(num_cluster):
            left_coeff_matrix = left_coeff_matrix + gamma[i][k] * inv_sigma[k]
            right_side = right_side + (gamma[i][k] * np.matmul(means[k], inv_sigma[:][:][k]))

        left_coeff_matrix = left_coeff_matrix + rho * np.eye(hdim)
        Y[i][:] = np.matmul(right_side, np.linalg.inv(left_coeff_matrix))


class Clustering:

    def __init__(self, autoencoder):
        self.ae = autoencoder

    def deep_ae_kmeans(self, data, test_data, clusters=2):
        """
        Compresses the input data and executes the K-Means clustering algorithm.
        :param data: The learning data subset.
        :param test_data: The testing data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        self.ae.fit()

        compressed_data = self.ae.encoder.predict(test_data)

        print(TAG, 'Clustering', len(test_data), 'data')
        return KMeans(clusters).fit_predict(compressed_data)

    def deep_cluster_kmeans(self, data, clusters):
        """
        Executes the DC-KMeans clustering algorithm.
        :param data: The learning data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        labels = []
        Y = []

        U = np.zeros((len(data), self.ae.encoding_dim))

        self.ae.model.compile(optimizer=self.ae.optimizer, loss='mse', metrics=['mae'])

        for i in range(self.ae.epochs):

            self.ae.model.fit(self.ae.data, self.ae.data, epochs=1, batch_size=self.ae.batch_size, shuffle=True)

            compressed_data = self.ae.encoder.predict(data)

            if len(Y) == 0:
                Y = np.copy(compressed_data)

            print(TAG, 'Clustering', len(Y), 'data')
            kmeans = KMeans(clusters).fit(Y)
            labels = kmeans.labels_

            # vis = Visualization(compressed_data, labels)
            # vis.reduce_scatter_plot()

            Y = update_Y(Y, U, compressed_data, self.ae.rho, self.ae.lbda, self.ae.gamma, self.ae.construct_quality,
                         kmeans.cluster_centers_, labels)

            U = U + Y - compressed_data

            print(i + 1, '/', self.ae.epochs)

        return labels

    def deep_ae_gmm(self, data, test_data, clusters):
        """
        Compresses the input data and executes the GMM clustering algorithm.
        :param data: The learning data subset.
        :param test_data: The testing data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        self.ae.fit()

        compressed_data = self.ae.encoder.predict(test_data)

        print(TAG, 'Clustering', len(test_data), 'data')
        return GaussianMixture(n_components=clusters, covariance_type='full').fit_predict(compressed_data)

    def deep_cluster_gmm(self, data, clusters):
        """
        Executes the DC-GMM clustering algorithm.
        :param data: The learning data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        labels = []
        Y = []

        U = np.zeros((len(data), self.ae.encoding_dim))

        self.ae.model.compile(optimizer=self.ae.optimizer, loss='mse', metrics=['mae'])

        for i in range(self.ae.epochs):

            self.ae.model.fit(self.ae.data, self.ae.data, epochs=1, batch_size=self.ae.batch_size, shuffle=True)

            compressed_data = self.ae.encoder.predict(data)

            if len(Y) == 0:
                Y = np.copy(compressed_data)

            print(TAG, 'Clustering', len(Y), 'data')
            gmm = GaussianMixture(n_components=clusters, covariance_type='full').fit(Y)
            labels = gmm.predict(Y)

            means = gmm.means_
            sigmas = gmm.covariances_

            # Y = update_Y(Y, U, compressed_data, self.rho, self.lbda, kmeans.cluster_centers_, labels)

            update_Y_GMM(Y, compressed_data, U, means, self.ae.rho, self.ae.lbda, self.ae.gamma,
                         self.ae.construct_quality, sigmas)

            U = U + Y - compressed_data
            # print(U)

            print(i + 1, '/', self.ae.epochs)

        return labels

    def deep_ae_dbscan(self, data, test_data, eps, min_points):
        """
        Compresses the input data and executes the DBSCAN clustering algorithm.
        :param data: The learning data subset.
        :param test_data: The testing data subset.
        :param eps: The radius DBSCAN uses to classify points as core, border or noise.
        :param min_points: The minimum number of points DBSCAN uses to determine neighborhoods.
        :return: A trained DAE and list of predictions.
        """

        self.ae.fit()

        compressed_data = self.ae.encoder.predict(test_data)

        print(TAG, 'Clustering', len(test_data), 'data')
        return DBSCAN(algorithm='auto', eps=eps, min_samples=min_points).fit_predict(compressed_data)

    def deep_cluster_dbscan(self, data, eps, min_points, clusters):
        """
        Executes the DC-DBSCAN clustering algorithm.
        :param data: The learning data subset.
        :param eps: The radius DBSCAN uses to classify points as core, border or noise.
        :param min_points: The minimum number of points DBSCAN uses to determine neighborhoods.
        :return: A trained DAE and list of predictions.
        """

        labels = []
        Y = []

        U = np.zeros((len(data), self.ae.encoding_dim))

        self.ae.model.compile(optimizer=self.ae.optimizer, loss='mse', metrics=['mae'])

        for i in range(self.ae.epochs):

            self.ae.model.fit(self.ae.data, self.ae.data, epochs=1, batch_size=self.ae.batch_size, shuffle=True)

            compressed_data = self.ae.encoder.predict(data)

            if len(Y) == 0:
                Y = np.copy(compressed_data)

            dbscan = DBSCAN(eps=eps, min_samples=min_points).fit(Y)
            labels = dbscan.labels_
            # data_rec = self.ae.decoder.predict(Y)

            c_means = get_cluster_means(compressed_data, labels)

            if len(c_means) == 0:
                c_means = np.zeros((clusters, len(compressed_data[0])))

            Y = update_Y(Y, U, compressed_data, self.ae.rho, self.ae.lbda, self.ae.gamma, self.ae.construct_quality,
                       c_means, labels)
            U = U + Y - compressed_data

            print(i + 1, '/', self.ae.epochs)

        return labels

    def deep_ae_hierarchical(self, data, test_data, clusters):
        """
        Compresses the input data and executes the hierarchical clustering algorithm.
        :param data: The learning data subset.
        :param test_data: The testing data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        self.ae.fit()

        compressed_data = self.ae.encoder.predict(test_data)

        print(TAG, 'Clustering', len(test_data), 'data')
        return AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit_predict(
            compressed_data)

    def deep_cluster_hierarchical(self, data, clusters):
        """
        Executes the DC-Hierarchical clustering algorithm.
        :param data: The learning data subset.
        :param clusters: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        labels = []
        Y = []

        U = np.zeros((len(data), self.ae.encoding_dim))

        self.ae.model.compile(optimizer=self.ae.optimizer, loss='mse', metrics=['mae'])

        for i in range(self.ae.epochs):

            self.ae.model.fit(self.ae.data, self.ae.data, epochs=1, batch_size=self.ae.batch_size, shuffle=True)

            compressed_data = self.ae.encoder.predict(data)

            if len(Y) == 0:
                Y = np.copy(compressed_data)

            print(TAG, 'Clustering', len(Y), 'data')
            hierarchical = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=clusters).fit(Y)
            labels = hierarchical.labels_
            # data_rec = self.ae.decoder.predict(Y)

            Y = update_Y(Y, U, compressed_data, self.ae.rho, self.ae.lbda, self.ae.gamma, self.ae.construct_quality,
                         get_cluster_means(compressed_data, labels), labels)
            U = U + Y - compressed_data
            # print(U)

            print(i + 1, '/', self.ae.epochs)

        return labels

    def deep_ae_ensemble(self, data, test_data, n):
        """
        Compresses the input data and executes all above clustering algorithms and combines the results with majority voting.
        :param data: The learning data subset.
        :param test_data: The testing data subset.
        :param n: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        self.ae.fit()

        compressed_data = self.ae.encoder.predict(test_data)

        print(TAG, 'Clustering', len(test_data), 'data')
        hierarchical = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=n).fit_predict(
            compressed_data)
        gmm = GaussianMixture(n_components=n, covariance_type='full').fit_predict(compressed_data)
        kmeans = KMeans(n).fit_predict(compressed_data)

        labels_list = [hierarchical, gmm, kmeans]

        return majority_vote(labels_list)

    def deep_cluster_ensemble(self, data, n):
        """
        Executes the DC-Ensemble clustering algorithm.
        :param data: The learning data subset.
        :param n: Number of clusters we want the data to be distributed into.
        :return: A trained DAE and list of predictions.
        """

        labels = []
        Y = []

        U = np.zeros((len(data), self.ae.encoding_dim))

        self.ae.model.compile(optimizer=self.ae.optimizer, loss='mse', metrics=['mae'])

        for i in range(self.ae.epochs):

            self.ae.model.fit(self.ae.data, self.ae.data, epochs=1, batch_size=self.ae.batch_size, shuffle=True)

            compressed_data = self.ae.encoder.predict(data)

            if len(Y) == 0:
                Y = np.copy(compressed_data)

            print(TAG, 'Clustering', len(Y), 'data')
            hierarchical = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters=n).fit_predict(Y)
            gmm = GaussianMixture(n_components=n, covariance_type='full').fit_predict(Y)
            kmeans = KMeans(n).fit_predict(Y)

            labels_list = [hierarchical, gmm, kmeans]

            labels = majority_vote(labels_list)

            c_means = get_cluster_means(compressed_data, labels)
            
            if len(c_means) < n:
                for _ in range(n - len(c_means)):
                    pad =  np.zeros(len(c_means[0]))
                    pad = pad.reshape((1, len(c_means[0])))
                    c_means = np.concatenate( (c_means,pad), axis=0 )
            

            Y = update_Y(Y, U, compressed_data, self.ae.rho, self.ae.lbda, self.ae.gamma, self.ae.construct_quality,
                    c_means, labels)
            U = U + Y - compressed_data

            print(i + 1, '/', self.ae.epochs)

        return labels
