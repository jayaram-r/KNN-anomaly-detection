"""
Methods for (local) intrinsic dimension estimation based on nearest neighbor distances.

"""
from __future__ import division
import numpy as np
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
    lid_est = -1.0 / np.mean(log_dist_ratio, axis=1)

    return lid_est
