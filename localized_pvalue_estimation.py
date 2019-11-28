"""
Localized p-value estimation methods for anomaly detection.

1. K-LPE method:
Zhao, Manqi, and Venkatesh Saligrama. "Anomaly detection with score functions based on nearest neighbor graphs."
Advances in neural information processing systems. 2009.

2. Averaged K-LPE method:
Qian, Jing, and Venkatesh Saligrama. "New statistic in p-value estimation for anomaly detection."
IEEE Statistical Signal Processing Workshop (SSP). IEEE, 2012.

"""
from __future__ import division
import numpy as np
import sys
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from metrics_custom import (
    distance_SNN,
    neighborhood_membership_vectors
)
import warnings
from numba import NumbaPendingDeprecationWarning

# Suppress numba warnings
warnings.filterwarnings('ignore', '', NumbaPendingDeprecationWarning)


class averaged_KLPE_anomaly_detection:
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        """

        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param metric: string or a callable that specifies the distance metric.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance.
                                         This is a secondary distance metric that is found to be better suited to
                                         high dimensional data.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. This is recommended when the number of points is
                                         large and/or when the dimension of the data is high.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.seed_rng = seed_rng

        self.num_samples = None
        self.neighborhood_range = None
        self.index_knn = None
        self.dist_stat_nominal = None
        np.random.seed(self.seed_rng)

    def fit(self, data):
        """
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :return: None
        """
        N, d = data.shape
        self.num_samples = N
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        # The distance statistic is averaged over this neighborhood range
        low = self.n_neighbors - int(np.floor(0.5 * (self.n_neighbors - 1)))
        high = self.n_neighbors + int(np.floor(0.5 * self.n_neighbors))
        self.neighborhood_range = (low, high)

        # Build the KNN graphs
        self.index_knn = self.build_knn_graphs(data)

        # Compute the distance statistic for every data point
        self.dist_stat_nominal = self.distance_statistic(data, exclude_self=True)

    def score(self, data_test, exclude_self=False):
        """
        Calculate the anomaly score (p-value) for a given test data set.

        :param data_test: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the
                          number of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return score: numpy array of shape `(n, 1)` containing the score for each point. Points with higher score
                       are more likely to be anomalous.
        """
        dist_stat_test = self.distance_statistic(data_test, exclude_self=exclude_self)
        score = ((1. / self.dist_stat_nominal.shape[0]) *
                 np.sum(dist_stat_test[:, np.newaxis] <= self.dist_stat_nominal[np.newaxis, :], axis=1))

        # Negative log of the p-value is returned as the anomaly score
        return -1.0 * np.log(np.clip(score, sys.float_info.min, None))

    def build_knn_graphs(self, data, min_n_neighbors=20, rho=0.5):
        """
        Build a KNN index for the given data set.
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param min_n_neighbors: minimum number of nearest neighbors to use for the `NN-descent` method.
        :param rho: `rho` parameter used by the `NN-descent` method.
        :return: A list with one or two KNN indices.
        """
        N, d = data.shape
        if self.approx_nearest_neighbors:
            # Construct an approximate nearest neighbor (ANN) index to query nearest neighbors
            params = {
                'metric': self.metric,
                'metric_kwds': self.metric_kwargs,
                'n_neighbors': max(1 + self.neighborhood_range[1], min_n_neighbors),
                'rho': rho,
                'random_state': self.seed_rng,
                'n_jobs': self.n_jobs
            }
            index_knn_primary = NNDescent(data, **params)
        else:
            # Construct the exact KNN graph
            index_knn_primary = NearestNeighbors(
                n_neighbors=(1 + self.neighborhood_range[1]),
                algorithm='brute',
                metric=self.metric,
                metric_params=self.metric_kwargs,
                n_jobs=self.n_jobs
            )
            index_knn_primary.fit(data)

        if self.shared_nearest_neighbors:
            # Query the `self.n_neighbors` nearest neighbors of each point.
            # Since each point will be selected as its own nearest neighbor, we query for `self.n_neighbors + 1`
            # neighbors and ignore the self neighbors
            nn_indices, _ = self.query_wrapper(data, index_knn_primary, self.n_neighbors + 1)

            # Create the neighborhood membership vector for each point.
            # `data_neighbors` will be numpy array of 0s and 1s, with shape `(N, N)` and dtype `np.uint8`
            data_neighbors = neighborhood_membership_vectors(nn_indices, N)

            # Set the diagonal elements of `data_neighbors` to 0 because we don't want a point to be in its
            # own neighborhood set
            np.fill_diagonal(data_neighbors, 0)

            # Construct another KNN index for the binary membership data using the shared nearest neighbor
            # distance metric
            if self.approx_nearest_neighbors:
                params = {
                    'metric': distance_SNN,
                    'n_neighbors': max(1 + self.neighborhood_range[1], min_n_neighbors),
                    'rho': rho,
                    'random_state': self.seed_rng,
                    'n_jobs': self.n_jobs
                }
                index_knn_secondary = NNDescent(data_neighbors, **params)
            else:
                index_knn_secondary = NearestNeighbors(
                    n_neighbors=(1 + self.neighborhood_range[1]),
                    algorithm='brute',
                    metric=distance_SNN,
                    n_jobs=self.n_jobs
                )
                index_knn_secondary.fit(data_neighbors)

            index_knn = [index_knn_primary, index_knn_secondary]
        else:
            index_knn = [index_knn_primary]

        return index_knn

    def distance_statistic(self, data, exclude_self=False):
        """
        Calculate the average distance statistic by querying the nearest neighbors of the given set of points.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return dist_stat: numpy array of distance statistic for each point.
        """
        if exclude_self:
            # Query an extra neighbor because the points are part of the KNN graph
            k1 = self.neighborhood_range[0] + 1
            k2 = self.neighborhood_range[1] + 1
            k = self.n_neighbors + 1
        else:
            k1, k2 = self.neighborhood_range
            k = self.n_neighbors

        if self.shared_nearest_neighbors:
            # Query the `k` nearest neighbors
            nn_indices_, _ = self.query_wrapper(data, self.index_knn[0], k)

            # Create the neighborhood membership vector for each point
            data_neighbors = neighborhood_membership_vectors(nn_indices_, self.num_samples)

            if exclude_self:
                # Set the diagonal elements of `data_neighbors` to 0 because we don't want a point to be in its
                # own neighborhood set
                np.fill_diagonal(data_neighbors, 0)

            nn_indices, nn_distances = self.query_wrapper(data_neighbors, self.index_knn[1], k2)
        else:
            nn_indices, nn_distances = self.query_wrapper(data, self.index_knn[0], k2)

        dist_stat = np.mean(nn_distances[:, (k1 - 1):], axis=1)
        return dist_stat

    def query_wrapper(self, data, index, k):
        """
        Unified wrapper for querying both the approximate and the exact KNN index.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param index: KNN index.
        :param k: number of nearest neighbors to query.
        :return:
        """
        if self.approx_nearest_neighbors:
            nn_indices, nn_distances = index.query(data, k=k)
        else:
            nn_distances, nn_indices = index.kneighbors(data, n_neighbors=k)

        return nn_indices, nn_distances
