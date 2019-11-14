"""
Generate data that follows a mixture of factor analyzers (MFA) model
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MFA_model:
    def __init__(self, n_components, dim, dim_latent=None, dim_latent_range=None, seed_rng=123):
        """
        Specify one of the inputs `dim_latent` or `dim_latent_range`, not both.

        :param n_components: (int) number of components in the MFA model. Value should be >= 1.
        :param dim: (int) dimension or number of features. Should be >= 2.
        :param dim_latent: (int or list/tuple/array of ints) dimension of the latent space per component.
                           If a single integer value is specified, that value is used for all the components.
                           The size of the list/tuple/array should be equal to `n_components`.
        :param dim_latent_range: tuple or list with two integer values specifying the range of the dimension of
                                 the latent space. For each component, the latent dimension is randomly selected
                                 from this range of values. Make sure the upper end of this range is smaller than
                                 `dim`.
        :param seed_rng: None or an integer to seed the random number generator.
        """
        self.n_components = n_components
        if n_components < 1:
            raise ValueError("Invalid value '{}' for 'n_components'".format(n_components))

        self.dim = dim
        if dim < 2:
            raise ValueError("Invalid value '{}' for 'dim'".format(dim))

        self.seed_rng = seed_rng
        np.random.seed(seed_rng)

        if dim_latent:
            if isinstance(dim_latent, int):
                self.dim_latent = [dim_latent] * self.n_components
            else:
                self.dim_latent = dim_latent
                if len(dim_latent) != self.n_components:
                    raise ValueError("Invalid value '{}' for 'dim_latent'".format(dim_latent))

        else:
            a, b = dim_latent_range
            if b >= self.dim:
                logger.warning("Range of the latent dimension should be smaller than {:d}".format(self.dim))
                b = self.dim - 1
                a = min(a, b)

            # Number of latent features is sampled uniformly from the range
            self.dim_latent = np.random.randint(a, high=(b + 1), size=self.n_components)

        self.parameters = self.generate_params()

    def generate_params(self):
        """
        Following the Bayesian conjugate priors model for MFA from
        http://mlg.eng.cam.ac.uk/zoubin/papers/nips99.pdf

        :return: dict of parameters.
        """
        self.parameters = dict()

        # Mixture weights are sampled from a symmetric Dirichlet distribution
        alpha = 2 * np.ones(self.n_components)
        self.parameters['weights'] = np.random.dirichlet(alpha)

        # Diagonal sensor noise covariance matrix that is shared across the components. The inverse of each diagonal
        # element of this matrix is independently sampled from a Gamma density
        # First sample the shape parameter (of the Gamma density) uniformly at random from a predefined interval
        k = np.random.uniform(low=1., high=5., size=self.dim)
        self.parameters['covariance_noise'] = np.diag(1. / np.random.gamma(k))

        # Mean vector for each component is sampled from a multivariate Gaussian distribution.
        # The factor loading matrices for each component are sampled as follows. Each column of a factor loading
        # matrix is sampled from a multivariate Gaussian with 0 mean vector and diagonal covariance matrix. The
        # inverse of each diagonal element (precision) of the covariance matrix is sampled from a Gamma density.
        means = []
        factor_loading_mats = []
        covariance_mats = []
        for i in range(self.n_components):
            # Component mean
            m = np.random.normal(loc=0., scale=5., size=self.dim)
            means.append(
                np.random.multivariate_normal(m, np.eye(self.dim))
            )
            # Component factor loading matrix
            mat = np.zeros((self.dim, self.dim_latent[i]))
            for j in range(self.dim_latent[i]):
                # For each column of the factor loading matrix
                k = np.random.uniform(low=1., high=5., size=self.dim)
                mat[:, j] = np.random.multivariate_normal(np.zeros(self.dim), np.diag(1. / np.random.gamma(k)))

            factor_loading_mats.append(mat)
            # Covariance matrix of the observed data conditioned on this component
            covariance_mats.append(
                np.dot(mat, np.transpose(mat)) + self.parameters['covariance_noise']
            )

        self.parameters['means'] = means
        self.parameters['factor_loading_mats'] = factor_loading_mats
        self.parameters['covariance_mats'] = covariance_mats

        return self.parameters

    def generate_data(self, N):
        """
        Generate `N` data samples from the model.

        :param N: (int) number of data samples to generate.
        :return: (data, labels) where
            - data: numpy array of shape `(N, self.dim)`.
            - labels: numpy array of component index labels of shape `(N, )`.
        """
        # Number of samples from each component
        counts = np.random.multinomial(N, self.parameters['weights'])
        data = []
        labels = []
        for i in range(self.n_components):
            if counts[i] < 1:
                continue

            data.append(np.random.multivariate_normal(self.parameters['means'][i],
                                                      self.parameters['covariance_mats'][i], counts[i]))
            labels.append(i * np.ones(counts[i], dtype=np.int))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels)
        # Randomize the order of the samples
        ind = np.random.permutation(N)

        return data[ind, :], labels[ind]
