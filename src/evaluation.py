from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import DataFrame, Series
from collections import Counter
from math import floor
from .knn import KNN
import numpy as np


def create_folds(y: Series, n_folds: int, random_state: np.random.RandomState) -> list:
    """
    Create stratified k folds.

    :param y: (Series) Data to create the folds from.
    :param n_folds: (int) Number of folds.
    :param random_state: (np.random.RandomState) RandomState to create random folds.
    :return: (list) List of folds.
    """
    fold_size = floor(y.size / n_folds)

    classes_count = Counter(y)
    classes_percent = {cls: count / y.size for cls, count in classes_count.items()}
    classes_size_in_fold = {cls: floor(classes_percent[cls] * fold_size) for cls in classes_count.keys()}
    classes_instances = {cls: {i for i in y.index.values if y[i] == cls} for cls in classes_count.keys()}

    folds = []
    instance_count = 0
    for _ in range(n_folds):
        fold = []

        for cls in classes_count.keys():
            class_fold = random_state.choice(list(classes_instances[cls]), classes_size_in_fold[cls], replace=False)
            classes_instances[cls] = classes_instances[cls].difference(class_fold)  # update available instances
            instance_count += len(class_fold)
            fold += list(class_fold)

        folds.append(fold)

    # Add remaining instances
    fold_i = 0
    for cls, instances in classes_instances.items():
        for i in instances:
            folds[fold_i].append(i)
            fold_i = (fold_i + 1) % n_folds

    return folds


def cross_validate(estimator: KNN, x: DataFrame, y: Series, folds: list) -> dict:
    """
    Run a k-fold cross-validation.

    :param estimator: (KNN) a KNN instance with a fit and a predict method.
    :param x: (DataFrame) Training instances.
    :param y: (Series) Training classes.
    :param folds: (int) List of folds/list of groups of instances ids
    :return: (dict) Dictionary with metrics for each fold. Metrics: accuracy, precision, recall, f1
    """
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    labels = list(set(y))
    n_labels = len(labels)

    for i, test_fold in enumerate(folds):
        train_folds = folds[:i] + folds[i+1:]
        train_fold = [index for fold in train_folds for index in fold]

        train_x = x.loc[train_fold, :]
        train_y = y[train_fold]

        test_x = x.loc[test_fold, :]
        test_y = y[test_fold]

        estimator.fit(train_x, train_y)
        pred_y = [estimator.predict(test_x.iloc[i, :]) for i in range(test_y.size)]

        accuracy = accuracy_score(test_y, pred_y)
        if n_labels == 2:
            precision = precision_score(test_y, pred_y)
            recall = recall_score(test_y, pred_y)
            f1 = f1_score(test_y, pred_y)
        else:
            precision = precision_score(test_y, pred_y, labels=labels, average='micro')
            recall = recall_score(test_y, pred_y, labels=labels, average='micro')
            f1 = f1_score(test_y, pred_y, labels=labels, average='micro')

        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)

    return results

