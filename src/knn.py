from pandas import DataFrame
from math import pow, sqrt
import numpy as np


class KNN:

    def __init__(self, k: int, dist: str = 'euclidean'):
        """
        Implement the K Nearest Neighbors.

        :param k: (int) Number of neighbors to consider on classification. Must be greater than 1.
        :param dist: (str, default 'euclidean') Distance metric. Possible values: euclidean, (TBD)...

        Raise ValueError if k <= 1.
        Raise ValueError if distance metric is unknown.
        """
        if k <= 1:
            raise ValueError('k must be greater than 1.')

        if dist not in ['euclidean']:
            raise ValueError(f'dist must be euclidean, (TBD)... {dist} found.')

        self.distance_metric = eval(f'self._{dist}')
        self.distance_matrix = None
        self.targets = None
        self.k = k

    def fit(self, x, y):
        """
        Create a distance matrix N x N, where N is the number of instances.

        :param x: (pandas.DataFrame or numpy.ndarray) Training instances.
        :param y: (pandas.Series or numpy.array) Target class for each instance.
        """
        if type(x) is DataFrame:
            x = x.to_numpy()

        n, m = x.shape
        self.targets = y
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                self.distance_matrix[i, j] = self.distance_metric(x[i], x[j])

    def predict(self, x):
        """

        :param x:
        :return:

        Raise ValueError if called before fit.
        """
        if self.distance_matrix is None:
            raise ValueError('You should fit the model before predicting.')

        raise NotImplementedError()

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """
        Get euclidean distance between two vector (a and b).

        :param a: (numpy.ndarray) Instance 1 or vector 1.
        :param b: (numpy.ndarray) Instance 2 or vector 2.

        :return: (float) Euclidean distance between a and b.
        """
        if a.size != b.size:
            raise ValueError(f'a and b must have equal lengths. {a.size} != {b.size}.')

        squared_error = [pow(a[i] - b[i], 2) for i in range(a.size)]
        sum_squared_error = sum(squared_error)
        sqrt_sum_squared_error = sqrt(sum_squared_error)

        return sqrt_sum_squared_error
