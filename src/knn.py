from pandas import DataFrame, Series
from collections import defaultdict
from collections import Counter
from math import pow, sqrt
import numpy as np

cache = {}

class KNN:

    def __init__(self, k: int, dist: str = 'euclidean', evaluator_method: str = 'majority', seed: int = 1234):
        """
        Implement the K Nearest Neighbors.

        :param k: (int) Number of neighbors to consider on classification. Must be greater than 1.
        :param dist: (str, default 'euclidean') Distance metric. Possible values: euclidean.
        :param evaluator_method: (str, default 'majority') Method of evaluating the nearest k neighbors.
            Possible values: majority, inverse_square, averaged_inverse_square.
        :param seed: (int, default 1234) Seed for random state.

        Raise ValueError if k <= 1.
        Raise ValueError if distance metric is unknown.
        """
        if k <= 1:
            raise ValueError('k must be greater than 1.')

        if dist not in ['euclidean']:
            raise ValueError(f'dist must be euclidean... {dist} found.')

        if evaluator_method not in ['majority', 'inverse_square', 'averaged_inverse_square']:
            raise ValueError(f'evaluator_method must be majority, inverse_square... {evaluator_method} found.')

        self.distance_metric = eval(f'self._{dist}')
        self.evaluator_method = eval(f'self._{evaluator_method}')
        self.random_state = np.random.RandomState(seed)
        self.training_instances = None
        self.training_instances_classes = None
        self.k = k

    def fit(self, training_instances, training_instances_classes):
        """
        Store "training_instances" and "training_instances_classes".

        :param training_instances: (pandas.DataFrame or numpy.ndarray) Training instances.
        :param training_instances_classes: (pandas.Series or numpy.array) Target class for each instance.
        """
        if type(training_instances) is DataFrame:
            training_instances = training_instances.to_numpy()

        self.training_instances = training_instances
        self.training_instances_classes = training_instances_classes

    def predict(self, testing_instance: any) -> any:
        """
        Predict the class of the instance "testing_instance".

        :param testing_instance: (Series/np.ndarray) Instance to be predicted.
        :return: (int/str) Target class.

        Raise ValueError if called before fit.
        """
        if self.training_instances is None:
            raise ValueError('You should fit the model before predicting.')

        if type(testing_instance) is Series:
            testing_instance = testing_instance.to_numpy()

        distances = [self.distance_metric(training_instance, testing_instance) for training_instance in self.training_instances]
        nearest_neighbors_ids = np.argsort(distances)
        nearest_k_neighbors_ids = nearest_neighbors_ids[:self.k]

        nearest_k_neighbors_distances = [distances[neighbor_id] for neighbor_id in nearest_k_neighbors_ids]
        nearest_k_neighbors_classes = [self.training_instances_classes.iloc[neighbor_id] for neighbor_id in nearest_k_neighbors_ids]

        return self.evaluator_method(nearest_k_neighbors_distances, nearest_k_neighbors_classes)

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """
        Get euclidean distance between two vector (a and b).

        :param a: (numpy.ndarray) Instance 1 or vector 1.
        :param b: (numpy.ndarray) Instance 2 or vector 2.

        :return: (float) Euclidean distance between a and b.
        """

        a_hash = a.tobytes()
        b_hash = b.tobytes()

        ab_hash = a_hash+b_hash if a_hash>b_hash else b_hash+a_hash

        if not ab_hash in cache.keys():
            if a.size != b.size:
                raise ValueError(f'a and b must have equal lengths. {a.size} != {b.size}.')

            squared_error = [pow(a[i] - b[i], 2) for i in range(a.size)]
            sum_squared_error = sum(squared_error)
            sqrt_sum_squared_error = sqrt(sum_squared_error)
            cache[ab_hash] = sqrt_sum_squared_error

        return cache[ab_hash]

    def _majority(self, x: np.ndarray, y: np.ndarray) -> any:
        """
        Give points to the classes to decide which one is the most likely.
        To do that, it considers a list of k nearest instances.
        Sums 1 point to each class in the list.
        Return the class with most points, tie-breaking with randomization.

        :param x: (np.ndarray) List of distances of the K nearest instances.
            Not used in this function. It exists only to keep the same interface with other methods.
        :param y: (np.ndarray) List of classes of the K nearest instances.
        :return: (int/str) Target class.
        """
        # Dictionary that returns 0 if key invalid.
        counter = defaultdict(lambda: 0)

        for x_class in y:
            counter[x_class] += 1

        return self._get_class_with_biggest_score(counter)

    def _inverse_square(self, x: np.ndarray, y: np.ndarray):
        """
        Give points to the classes to decide which one is the most likely.
        To do that, it considers a list of k nearest instances.
        Sums 1/(distance^2) points to each class in the list.
        Return the class with most points, tie-breaking with randomization.
        An expection is if some distances are zero, in this case, it is used majority over the instances at distance 0.

        :param x: (np.ndarray) List of distances of the K nearest instances.
        :param y: (np.ndarray) List of classes of the K nearest instances.
        :return: (int/str) Target class.
        """
        # If at least one instance is at distance 0, we must use majority over these instances that are at distance 0.
        if 0 in x:
            new_x_dists = []
            new_y = []

            for x_dist, x_class in zip(x, y):
                if x_dist == 0:
                    new_x_dists.append(x_dist)
                    new_y.append(x_class)

            new_x_dists = np.array(new_x_dists)
            new_y = np.array(new_y)

            return self._majority(new_x_dists, new_y)

        # Otherwise, follows to the procedure.
        # Dictionary that returns 0 if key invalid.
        counter = defaultdict(lambda: 0)

        for x_dist, x_class in zip(x, y):
            counter[x_class] += 1 / pow(x_dist, 2)  # The weight is 1/d^2.

        return self._get_class_with_biggest_score(counter)

    def _averaged_inverse_square(self, x: np.ndarray, y: np.ndarray):
        """
        Same as "inverse_square", but divide the classes score by the number of training instances that they have.

        :param x: (np.ndarray) List of distances of the K nearest instances.
        :param y: (np.ndarray) List of classes of the K nearest instances.
        :return: (int/str) Target class.
        """
        # If at least one instance is at distance 0, we must use majority over these instances that are at distance 0.
        if 0 in x:
            new_x_dists = []
            new_y = []

            for x_dist, x_class in zip(x, y):
                if x_dist == 0:
                    new_x_dists.append(x_dist)
                    new_y.append(x_class)

            new_x_dists = np.array(new_x_dists)
            new_y = np.array(new_y)

            return self._majority(new_x_dists, new_y)

        # Otherwise, follows to the procedure.
        # Dictionary that returns 0 if key invalid.
        counter = defaultdict(lambda: 0)

        for x_dist, x_class in zip(x, y):
            counter[x_class] += 1 / pow(x_dist, 2) / self.training_instances_classes.to_list().count(x_class)

        return self._get_class_with_biggest_score(counter)

    def _get_class_with_biggest_score(self, class_scores: dict) -> any:
        """
        Given a score for each class, return the class with the biggest score.
        If tie, we choose a class randomly.

        :param class_scores: (dict) {class_label (int/str): score (int)}
        :return: (int/str) The class with the biggest score.
        """
        ordered_class_count = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)  # list of tuples
        max_frequency = ordered_class_count[0][1]
        classes_with_most_instances = []

        for instance_class, frequency in ordered_class_count:
            if frequency == max_frequency:
                classes_with_most_instances.append(instance_class)
            else:
                break

        # Return a random class with most instances
        return self.random_state.choice(classes_with_most_instances)
