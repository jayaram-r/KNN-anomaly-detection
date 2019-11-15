"""
Some custom distance metrics and similarity measures.
"""
from __future__ import division
import numpy as np
import numba
import math


@numba.njit(fastmath=True)
def distance_norm_3tensors(x, y, shape=None, p=2, q=2, r=2):
    """
    Distance between two 3rd order (rank 3) tensors under the specified type of norm.
    The tensors `x` and `y` should be flattened into 1D numpy arrays before calling this function.
    If `xt` and `yt` are the tensors, each of shape `shape`, they can be flattened into a 1D array using
    `x = xt.reshape(-1)` (and likewise for `yt`). The shape is passed as input, which is used by the function
    to reshape the arrays to tensors.

    The inputs are taken as 1d arrays in order to be consistent  with the function signature required by
    another library.

    The inputs `p, q, r` together specify the type of norm used. They should be integers >= 1.
    For example, the default values `p = q = r = 2` compute the `l2` or Euclidean norm of the flattened tensor.

    :param x: numpy array of shape `(n, )` with the first flattened tensor.
    :param y: numpy array of shape `(n, )` with the second flattened tensor.
    :param shape: tuple of three values specifying the shape of the tensors. Ideally this should be a required
                  input, but if it is not specified, `x` and `y` will be reshaped into `(1, 1, n)`.
    :param p: norm parameter which is an int value >= 1.
    :param q: same as `p`.
    :param r: same as `p`.

    :return: norm value which is a float.
    """
    if shape is None:
        shape = (1, 1, x.shape[0])

    zt = np.abs(x - y).reshape(shape)
    pow1 = q / r
    pow2 = p / q
    pow3 = 1. / p
    s = 0.
    for i in range(shape[0]):
        sj = 0.
        for j in range(shape[1]):
            sj += (np.sum(zt[i, j, :] ** r) ** pow1)

        s += (sj ** pow2)

    return s ** pow3


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
    :param shape: tuple of three values specifying the shape of the tensors. Ideally this should be a required
                  input, but if it is not specified, `x` and `y` will be reshaped into `(1, 1, n)`.

    :return: norm value which should be in the range [0, 1].
    """
    if shape is None:
        shape = (1, 1, x.shape[0])

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
