from pandas import DataFrame
import os


def save_results(dataset: str, results: DataFrame):
    """
    Save the results as a CSV file in the correct folder.

    :param dataset: (str) Dataset name.
    :param results: (DataFrame) Results to be saved.
    """
    os.makedirs(f'results', exist_ok=True)
    filepath = f'results/{dataset}.csv'
    results.to_csv(filepath, header=True, index=False)
