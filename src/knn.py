from collections import defaultdict
from random import choice
from pandas import DataFrame, Series
from math import pow, sqrt
import numpy as np


class KNN:

    def __init__(self, k: int, dist: str = 'euclidean', evaluator_method: str = 'majority'):
        """
        Implement the K Nearest Neighbors.

        :param k: (int) Number of neighbors to consider on classification. Must be greater than 1.
        :param dist: (str, default 'euclidean') Distance metric. Possible values: euclidean, (TBD)...
        :param evaluator_method: (str, default 'majority') Method of evaluating the nearest k neighbors. Possible values: majority, inverse_square, (TBD)...

        Raise ValueError if k <= 1.
        Raise ValueError if distance metric is unknown.
        """
        if k <= 1:
            raise ValueError('k must be greater than 1.')

        if dist not in ['euclidean']:
            raise ValueError(f'dist must be euclidean, (TBD)... {dist} found.')

        if dist not in ['majority']:
            raise ValueError(f'evaluator_method must be majority, (TBD)... {evaluator_method} found.')

        self.distance_metric = eval(f'self._{dist}')
        self.evaluator_method = eval(f'self._{evaluator_method}')
        self.x = None
        self.y = None
        self.k = k

    def fit(self, x, y):
        """
        Store x and y.

        :param x: (pandas.DataFrame or numpy.ndarray) Training instances.
        :param y: (pandas.Series or numpy.array) Target class for each instance.
        """
        if type(x) is DataFrame:
            x = x.to_numpy()

        self.x = x
        self.y = y

    def predict(self, x):
        """
        Predict the class of the instance x.

        :param x: (Series/np.ndarray) Instance to be predicted.
        :return: (int/str) Target class.

        Raise ValueError if called before fit.
        """
        if self.x is None:
            raise ValueError('You should fit the model before predicting.')

        if type(x) is Series:
            x = x.to_numpy()

        distances_by_id = {inst_id: self.distance_metric(inst, x) for inst_id, inst in enumerate(self.x)}

        from operator import itemgetter
        nearest_k_ids = list(map(itemgetter(0), sorted(distances_by_id.items(), key=itemgetter(1))))[:self.k]

        x_dists = [distances_by_id[id] for id in nearest_k_ids]
        y = [self.y[id] for id in nearest_k_ids]

        return self.evaluator_method(x_dists, y)

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

    def _majority(self, x_dists: np.ndarray, y: np.ndarray):
        """
        Give points to the classes to decide which one is the most likely.
        To do that, it considers a list of k nearest instances.
        Sums 1 point to each class in the list.
        Return the class with most points, tie-breaking with randomization.
        ...

        :param x_dists: (np.ndarray) List of distances of the K nearest instances.
        :param y: (np.ndarray) List of classes of the K nearest instances.
        :return: (int/str) Target class.
        """

        counter = defaultdict(lambda: 0) # Dictionary that returns 0 if key invalid.

        for x_class in y:
            counter[x_class] += 1

        max_counter_value = max(counter.values()) # How many instances the class with more instances have.

        classes_with_most_instances = set()
        for x_class, class_counter_value in counter.items():
            if class_counter_value == max_counter_value:
                classes_with_most_instances.add(x_class)

        return choice(classes_with_most_instances) # Return a random class with most instance.

    def _inverse_square(self, x_dists: np.ndarray, y: np.ndarray):
        """
        Give points to the classes to decide which one is the most likely.
        To do that, it considers a list of k nearest instances.
        Sums 1/(distance^2) points to each class in the list.
        Return the class with most points, tie-breaking with randomization.
        An expection is if some distances are zero, in this case, it is used majority over the instances at distance 0.
        ...

        :param x_dists: (np.ndarray) List of distances of the K nearest instances.
        :param y: (np.ndarray) List of classes of the K nearest instances.
        :return: (int/str) Target class.
        """

        if 0 in x_dists: # If at least one instance is at distance 0, we must use majority over these instances that are at distance 0.
            new_x_dists = []
            new_y = []
            for x_dist, x_class in zip(x_dists, y):
                if x_dist == 0:
                    new_x_dists.append(x_dist)
                    new_y.append(x_class)
            return self._majority(new_x_dists, new_y)

        # Otherwise, follows to the procedure.

        counter = defaultdict(lambda: 0) # Dictionary that returns 0 if key invalid.

        for x_dist, x_class in zip(x_dists, y):
            counter[x_class] += 1 / pow(x_dist, 2) # The weight is 1/d^2.

        max_counter_value = max(counter.values()) # How many ''points'' the class with more ''points'' have.

        classes_with_most_instances = set()
        for x_class, class_counter_value in counter.items():
            if class_counter_value == max_counter_value:
                classes_with_most_instances.add(x_class)

        return choice(classes_with_most_instances) # Return a random class with most ''points''.
