"""
Methods for (local) intrinsic dimension estimation based on nearest neighbor distances.

"""
from __future__ import division
import numpy as np
from scipy import stats
import sys


def lid_mle_amsaleg(knn_distances):
    """
    Local intrinsic dimension (LID) estimators from the papers,
    1. Amsaleg, Laurent, et al. "Estimating local intrinsic dimensionality." Proceedings of the 21th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining. ACM, 2015.

    2. Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality."
    arXiv preprint arXiv:1801.02613 (2018).

    :param knn_distances: numpy array of k nearest neighbor distances. Has shape `(n, k)` where `n` is the
                          number of points and `k` is the number of neighbors.
    :return: `lid_est` is a numpy array of shape `(n, )` with the local intrinsic dimension estimates
              in the neighborhood of each point.
    """
    n, k = knn_distances.shape
    # Replace 0 distances with a very small float value
    knn_distances = np.clip(knn_distances, sys.float_info.min, None)
    log_dist_ratio = np.log(knn_distances / knn_distances[:, -1].reshape((n, 1)))
    # lid_est = -k / np.sum(log_dist_ratio, axis=1)
    lid_est = -(k - 1) / np.sum(log_dist_ratio, axis=1)

    return lid_est


def id_two_nearest_neighbors(knn_distances):
    """
    Estimate the intrinsic dimension of the data using the Two-nearest-neighbor method proposed in the following
    paper:
    Facco, Elena, et al. "Estimating the intrinsic dimension of datasets by a minimal neighborhood information."
    Scientific reports 7.1 (2017): 12140.

    :param knn_distances: numpy array of k nearest neighbor distances. Has shape `(n, k)` where `n` is the
                          number of points and `k` is the number of neighbors.
    :return: float value estimate of the intrinsic dimension.
    """
    n, k = knn_distances.shape
    # Ratio of 2nd to 1st nearest neighbor distances
    nn_ratio = knn_distances[:, 1] / np.clip(knn_distances[:, 0], sys.float_info.min, None)

    # Empirical CDF of `nn_ratio`
    nn_ratio_sorted = np.sort(nn_ratio)
    # Insert value 1 as the minimum value, which will have an empirical CDF of 0.
    # This will ensure that the line passes through the origin
    nn_ratio_sorted = np.insert(nn_ratio_sorted, 0, 1.0)
    ecdf = np.arange(n + 1) / float(n + 1)

    xs = np.log(nn_ratio_sorted)
    ys = -1 * np.log(1 - ecdf)
    # Fit a straight line. The slope of this line gives an estimate of the intrinsic dimension
    slope, intercept, _, _, _ = stats.linregress(xs, ys)

    return slope
