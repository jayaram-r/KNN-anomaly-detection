"""
Some custom distance metrics and similarity measures.
"""
from __future__ import division
import numpy as np
import numba


@numba.njit(fastmath=True)
def distance_norm_3tensors(x, y, shape=None, norm_type=(2, 2, 2)):
    """
    Distance between two 3rd order (rank 3) tensors under the specified type of norm.
    The tensors `x` and `y` should be flattened into 1D numpy arrays before calling this function.
    If `xt` and `yt` are the tensors, each of shape `shape`, they can be flattened into a 1D array using
    `x = xt.reshape(-1)` (and likewise for `yt`). The shape is passed as input, which is used by the function
    to reshape the arrays to tensors.

    The inputs are taken as 1d arrays in order to be consistent  with the function signature required by
    another library.

    The input `norm_type` is a tuple of length three that takes the norm parameters `p, q, r`, which together specify
    the type of norm used. They should be integers >= 1.
    For example, the default value `norm_type=(2, 2, 2)` computes the `l2` or Euclidean norm of the flattened tensor.
    The value `norm_type=(1, 1, 1)` computes the `l1` norm of the flattened tensor.

    The special case `p = -1` with finite `q` and `r` calculates the norm with `p = \infty`. The value `-1` is
    used instead of `np.infty` because numba does not handle tuple with mixed float and int values.

    :param x: numpy array of shape `(n, )` with the first flattened tensor.
    :param y: numpy array of shape `(n, )` with the second flattened tensor.
    :param shape: tuple of three values specifying the shape of the tensors. This is a required argument.
    :param norm_type: tuple of three values `(p, q, r)` that together define the type of norm to be used.
                      `q` and `r` should be integers >= 1. `p = -1` is a special value that calculates the
                      norm with `p = \infty`; otherwise `p` should be an integer >= 1.

    :return: distance value which is a non-negative float.
    """
    zt = np.abs(x - y).reshape(shape)
    # q / r
    pow1 = norm_type[1] / norm_type[2]
    if norm_type[0] > 0:
        # p / q
        pow2 = norm_type[0] / norm_type[1]
        s = 0.
        for i in range(shape[0]):
            sj = 0.
            for j in range(shape[1]):
                sj += (np.sum(zt[i, j, :] ** norm_type[2]) ** pow1)

            s += (sj ** pow2)

        # power 1 / p
        dist = s ** (1. / norm_type[0])
    else:
        # p = \infty
        s = -1.0
        for i in range(shape[0]):
            sj = 0.
            for j in range(shape[1]):
                sj += (np.sum(zt[i, j, :] ** norm_type[2]) ** pow1)

            if sj > s:
                s = sj

        dist = s ** (1. / norm_type[1])

    return dist


@numba.njit(fastmath=True)
def distance_angular_3tensors(x, y, shape=None):
    """
    Cosine angular distance between two 3rd order (rank 3) tensors.
    The tensors `x` and `y` should be flattened into 1D numpy arrays before calling this function.
    If `xt` and `yt` are the tensors, each of shape `shape`, they can be flattened into a 1D array using
    `x = xt.reshape(-1)` (and likewise for `yt`). The shape is passed as input, which is used by the function
    to reshape the arrays to tensors.

    The inputs are taken as 1d arrays in order to be consistent  with the function signature required by
    another library.

    :param x: numpy array of shape `(n, )` with the first flattened tensor.
    :param y: numpy array of shape `(n, )` with the second flattened tensor.
    :param shape: tuple of three values specifying the shape of the tensors. This is a required argument.

    :return: distance value which should be in the range [0, 1].
    """
    xt = x.reshape(shape)
    yt = y.reshape(shape)
    s = 0.
    for i in range(shape[0]):
        val1 = np.sum(xt[i, :, :] * yt[i, :, :])
        val2 = np.sum(xt[i, :, :] * xt[i, :, :]) ** 0.5
        val3 = np.sum(yt[i, :, :] * yt[i, :, :]) ** 0.5
        if val2 > 0. and val3 > 0.:
            s += (val1 / (val2 * val3))

    # Angular distance is the cosine-inverse of the average cosine similarity, divided by `pi` to normalize
    # the distance to the range `[0, 1]`
    s = max(-1., min(1., s / shape[0]))

    return np.arccos(s) / np.pi