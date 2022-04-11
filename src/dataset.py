from pandas import Series, DataFrame, concat
from pmlb import fetch_data
from statistics import mean, stdev


class Dataset:

    def __init__(self, name: str):
        """
        Represent a dataset from the PMLB library.
        Load the data into the x and y properties and preprocess data, if needed.

        :param name: (str) Dataset name. Used directly on fetch_data function.
            Possible values: ['iris', 'wine_quality_white', 'car_evaluation', 'churn', 'dis', 'breast_cancer', 'lupus', 'spambase']
        """
        possible_datasets = ['iris', 'wine_quality_white', 'car_evaluation', 'churn',
                             'dis', 'breast_cancer', 'lupus', 'spambase']

        if name not in possible_datasets:
            raise ValueError(f'Dataset "{name}" not available. Possible values: [\'iris\', \'wine_quality_white\', '
                             f'\'car_evaluation\', \'churn\', \'dis\', \'breast_cancer\', \'lupus\', \'spambase\']')

        data = fetch_data(name, dropna=True)
        data.drop_duplicates(inplace=True)

        self.x = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]
        self.n_folds = 0

        # Function called to preprocess the data if needed
        exec(f'self._{name}()')

    def _iris(self):
        """
        Classification of types of flowers regarding sepal and petal width and height.
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/iris/metadata.yaml

        n_observations: 150
        n_features: 4
        n_classes: 3
        imbalance: 0
        """
        self.n_folds = 5
        continuous_features = list(range(4))  # 0, 1, 2, 3
        self._normalize_features(continuous_features)

    def _wine_quality_white(self):
        """
        Features include 16 statistics that describe intentionally warped images of typed capital letters.
        The images are not included.
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/wine_quality_white/metadata.yaml

        n_observations: 4898
        n_features: 11
        n_classes: 7
        imbalance: 0.21
        """
        self.n_folds = 10
        continuous_features = list(range(11))  # 0, 1, 2, 3, ..., 10
        self._normalize_features(continuous_features)

    def _car_evaluation(self):
        """
        Mushroom records drawn from The Audubon Society Field Guide to North American Mushrooms.
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/car_evaluation/metadata.yaml

        n_observations: 1728
        n_features: 21
        n_classes: 4
        imbalance: 0.39
        """
        self.n_folds = 10

    def _dis(self):
        """
        ...
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/dis/metadata.yaml

        n_observations: 3772
        n_features: 29
        n_classes: 2
        imbalance: 0.94
        """
        self.n_folds = 10
        continuous_features = [0, 17, 19, 21, 23, 25]
        categorical_features = [1, 26, 27, 28]

        self._normalize_features(continuous_features)
        self._one_hot_encode_features(categorical_features)

    def _churn(self):
        """
        ...
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/churn/metadata.yaml

        n_observations: 5000
        n_features: 20
        n_classes: 2
        imbalance: 0.51
        """
        self.n_folds = 10
        continuous_features = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        categorical_features = [2, 19]

        self._normalize_features(continuous_features)
        self._one_hot_encode_features(categorical_features)

    def _breast_cancer(self):
        """
        ...
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/breast_cancer/metadata.yaml

        n_observations: 286
        n_features: 9
        n_classes: 2
        imbalance: 0.16
        """
        self.n_folds = 5
        continuous_features = [2]
        categorical_features = [0, 1, 3, 4, 5, 7]

        self._normalize_features(continuous_features)
        self._one_hot_encode_features(categorical_features)

    def _lupus(self):
        """
        A small dataset meant to compare the time from getting lupus to diagnosis against the time
        from diagnosis to death.
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/lupus/metadata.yaml

        n_observations: 86
        n_features: 3
        n_classes: 2
        imbalance: 0.04
        """
        self.n_folds = 5
        continuous_features = list(range(3))
        self._normalize_features(continuous_features)

    def _spambase(self):
        """
        ...
        Metadata: https://github.com/EpistasisLab/pmlb/blob/master/datasets/spambase/metadata.yaml

        n_observations: 4601
        n_features: 57
        n_classes: 2
        imbalance: 0.04
        """
        self.n_folds = 10
        continuous_features = list(range(57))
        self._normalize_features(continuous_features)

    @staticmethod
    def _normalize_feature(x: Series) -> Series:
        """
        Normalize feature using min/max approach.

        :param x: (Series) Feature values
        :return: (Series) Normalized feature.
        """
        min_x = x.min()
        max_x = x.max()
        normalized = (x - min_x) / (max_x - min_x)

        # mean_x = mean(x)
        # stdev_x = stdev(x)
        # normalized = (x - mean_x) / stdev_x

        return normalized

    @staticmethod
    def _one_hot_encode_feature(x: Series) -> DataFrame:
        """
        Convert each category from a categorical feature into a binary feature.

        :param x: (Series) Feature values.
        :return: (DataFrame) If N categories, returns a MxN binary matrix.
        """
        categories = set(x)
        columns = [f'{x.name}_{category}' for category in categories]
        result = DataFrame(columns=columns, index=x.index.values)

        paired_categories = zip(categories, columns)
        for j, paired_category in enumerate(paired_categories):
            category, column = paired_category
            result[column] = [int(x.iloc[i] == category) for i in range(x.size)]

        return result

    def _normalize_features(self, features: list):
        """
        Normalize a sequence of features. It will change self.x.

        :param features: (list) List of feature IDs to be normalized.
        """
        for f in features:
            self.x.iloc[:, f] = self._normalize_feature(self.x.iloc[:, f])

    def _one_hot_encode_features(self, features: list):
        """
        OneHot encode a sequence of features. It will the dimension of self.x.

        :param features: (list) List of feature IDs to be normalized.
        """
        result = None
        for f in features:
            dataframe = self._one_hot_encode_feature(self.x.iloc[:, f])
            result = concat([result, dataframe], axis=1, copy=False)

        self.x = self.x.drop(columns=self.x.columns[features])
        self.x = concat([self.x, result], axis=1, copy=False)
