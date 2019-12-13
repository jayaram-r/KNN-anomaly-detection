"""
Basic k nearest neighbors classifier that supports approximate nearest neighbor querying and custom distance
metrics including shared nearest neighbors.
"""
import numpy as np
from knn_index import KNNIndex
from numba import njit, int64, float64
from numba.types import Tuple


def wrapper_knn(data, labels, k,
                data_test=None, labels_test=None,
                metric='euclidean', metric_kwargs=None,
                shared_nearest_neighbors=False,
                approx_nearest_neighbors=True,
                n_jobs=1,
                seed_rng=123):
    knn_model = KNNClassifier(
        n_neighbors=k,
        metric=metric, metric_kwargs=metric_kwargs,
        shared_nearest_neighbors=shared_nearest_neighbors,
        approx_nearest_neighbors=approx_nearest_neighbors,
        n_jobs=n_jobs,
        seed_rng=seed_rng
    )
    knn_model.fit(data, labels)

    if data_test is None:
        labels_pred = knn_model.predict(data, is_train=True)
        # error rate
        mask = labels_pred != labels
        err_rate = float(mask[mask].shape[0]) / labels.shape[0]
    else:
        labels_pred = knn_model.predict(data_test, is_train=False)
        # error rate
        mask = labels_pred != labels_test
        err_rate = float(mask[mask].shape[0]) / labels_test.shape[0]

    return err_rate


@njit(Tuple((float64[:], float64[:, :]))(int64[:, :], int64[:], int64), fastmath=True)
def neighbors_label_counts(index_neighbors, labels_train, n_classes):
    """
    Given the index of neighboring samples from the training set and the labels of the training samples,
    find the label counts among the k neighbors and assign the label corresponding to the highest count as the
    prediction.

    :param index_neighbors: numpy array of shape `(n, k)` with the index of `k` neighbors of `n` samples.
    :param labels_train: numpy array of shape `(m, )` with the class labels of the `m` training samples.
    :param n_classes: (int) number of distinct classes.

    :return:
        - labels_pred: numpy array of shape `(n, )` with the predicted labels of the `n` samples. Needs to converted
                       to type `np.int` at the calling function. Numba is not very flexible.
        - counts: numpy array of shape `(n, n_classes)` with the count of each class among the `k` neighbors.
    """
    n, k = index_neighbors.shape
    counts = np.zeros((n, n_classes))
    labels_pred = np.zeros(n)
    for i in range(n):
        cnt_max = -1.
        ind_max = 0
        for j in range(k):
            c = labels_train[index_neighbors[i, j]]
            counts[i, c] += 1
            if counts[i, c] > cnt_max:
                cnt_max = counts[i, c]
                ind_max = c

        labels_pred[i] = ind_max

    return labels_pred, counts


class KNNClassifier:
    """
    Basic k nearest neighbors classifier that supports approximate nearest neighbor querying and custom distance
    metrics including shared nearest neighbors.
    """
    def __init__(self,
                 n_neighbors=1,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        """
        :param n_neighbors: int value specifying the number of nearest neighbors. Should be >= 1.
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
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.seed_rng = seed_rng

        self.index_knn = None
        self.labels_train = None
        self.n_classes = None
        self.label_enc = None
        self.label_dec = None

    def fit(self, X, y):
        """

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param y: numpy array of class labels of shape `(N, )`.
        :return:
        """
        label_set = np.unique(y)
        ind = np.arange(label_set.shape[0])
        self.n_classes = label_set.shape[0]
        # Mapping from label values to integers and its inverse
        d = dict(zip(label_set, ind))
        self.label_enc = np.vectorize(d.__getitem__)

        d = dict(zip(ind, label_set))
        self.label_dec = np.vectorize(d.__getitem__)

        self.labels_train = self.label_enc(y)
        self.index_knn = KNNIndex(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )

    def predict(self, X, is_train=False):
        """

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param is_train: Set to True if prediction is being done on the same data used to train.

        :return: numpy array with the class predictions, of shape `(N, )`.
        """
        # Get the indices of the nearest neighbors from the training set
        nn_indices, nn_distances = self.index_knn.query(X, k=self.n_neighbors, exclude_self=is_train)

        if self.n_neighbors > 1:
            labels_pred, counts = neighbors_label_counts(nn_indices, self.labels_train, self.n_classes)
            labels_pred = labels_pred.astype(np.int)
        else:
            labels_pred = self.labels_train[nn_indices[:, 0]]

        return self.label_dec(labels_pred)

    def predict_proba(self, X, is_train=False):
        """
        Estimate the probability of each class along with the predicted most-frequent class.

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param is_train: Set to True if prediction is being done on the same data used to train.

        :return:
            - numpy array with the class predictions, of shape `(N, )`.
            - numpy array with the estimated probability of each class, of shape `(N, self.n_classes)`.
              Each row should sum to 1.
        """
        # Get the indices of the nearest neighbors from the training set
        nn_indices, nn_distances = self.index_knn.query(X, k=self.n_neighbors, exclude_self=is_train)

        if self.n_neighbors > 1:
            labels_pred, counts = neighbors_label_counts(nn_indices, self.labels_train, self.n_classes)
            labels_pred = labels_pred.astype(np.int)
        else:
            labels_pred = self.labels_train[nn_indices[:, 0]]
            counts = np.ones(labels_pred.shape[0])

        proba = counts / self.n_neighbors
        return self.label_dec(labels_pred), proba

    def fit_predict(self, X, y):
        """

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param y: numpy array of class labels of shape `(N, )`.
        :return: numpy array with the class predictions, of shape `(N, )`.
        """
        self.fit(X, y)
        return self.predict(X, is_train=True)
