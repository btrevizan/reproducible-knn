from src.evaluation import create_folds, cross_validate
from pandas import DataFrame, concat
from src.utils import save_results
from src.dataset import Dataset
from src.knn import KNN
from fire import Fire
from math import sqrt
import numpy as np


class Main:

    def evaluate(self, dataset: str, seed: int = 1234):
        """
        Evaluate a model using the specified dataset and changing its parameters.
        We use k-fold cross validation repeated 5 times as the evaluation method.
        The results are saved in the results/<dataset>/<execution number>.csv

        :param dataset: (str) Name of the dataset to be used.
        :param seed: (int, default 1234) Seed for random state.
        """
        cols = ['dataset', 'cv', 'k_label', 'k', 'dist', 'evaluator', 'n_folds', 'fold', 'accuracy', 'precision', 'recall', 'f1']
        results = DataFrame(columns=cols)

        data = Dataset(dataset)
        n = (data.y.size / data.n_folds) * (data.n_folds - 1)

        possible_k = {'5': 5, 'sqrt(n)': int(sqrt(n)), 'n': int(n)}
        possible_dist = ['euclidean']
        possible_evaluator = ['majority', 'inverse_square']

        for cv in range(5):
            print(f'Cross validation #{cv}')

            rs = np.random.RandomState(seed)
            folds = create_folds(data.y, data.n_folds, rs)

            for k_label, k in possible_k.items():
                for dist in possible_dist:
                    for evaluator in possible_evaluator:
                        print(f'\tValidate k={k}, dist={dist}, evaluator={evaluator}...')

                        estimator = KNN(k, dist, evaluator)
                        metrics = cross_validate(estimator, data.x, data.y, folds)

                        print(f'\t\tAccuracy={np.mean(metrics["accuracy"])} +- {np.std(metrics["accuracy"])}')
                        print(f'\t\tPrecision={np.mean(metrics["precision"])} +- {np.std(metrics["precision"])}')
                        print(f'\t\tRecall={np.mean(metrics["recall"])} +- {np.std(metrics["recall"])}')
                        print(f'\t\tF1-Score={np.mean(metrics["f1"])} +- {np.std(metrics["f1"])}')

                        for i in range(data.n_folds):
                            result = DataFrame({
                                'dataset': [dataset],
                                'cv': [cv],
                                'k_label': [k_label],
                                'k': [k],
                                'dist': [dist],
                                'evaluator': [evaluator],
                                'n_folds': [data.n_folds],
                                'fold': [i],
                                'accuracy': [metrics['accuracy'][i]],
                                'precision': [metrics['precision'][i]],
                                'recall': [metrics['recall'][i]],
                                'f1': [metrics['f1'][i]]
                            })

                            results = concat([results, result], ignore_index=True)

        save_results(dataset, results)


if __name__ == '__main__':
    Fire(Main)
