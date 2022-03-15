from src.knn import KNN
from fire import Fire


class Main:

    def cv(self, dataset: str, k: int, dist: str = 'euclidean'):
        """
        Cross-validate a model.

        :param dataset: (str) Path to the dataset.
        :param k: (int) Number of neighbors to consider on classification. Must be greater than 1.
        :param dist: (str, default 'euclidean') Distance metric. Possible values: euclidean, (TBD)...
        """
        raise NotImplementedError()


if __name__ == '__main__':
    Fire(Main)
