# Reproducible KNN
The KNN algorithm is written in `src/knn.py`.
All functions have documentation that guides the user on how 
to use the code. However, we offer a command line interface to
ease the usage of the model.

## Dependencies
As a Python application, you can use the `requirements.txt`
file to install the application's dependencies:
```{shell}
$ pip install -r requirements.txt
```

## Usage
We have the following available methods in the command line:
- evaluate

### How to check the method usage
To check how to use a specific method, you can run:
```{shell}
$ python main.py <method> -h
```
This will show you the mandatory and optional parameters
as well a description of the method. For example:
```{shell}
$ python main.py evaluate -h
```
```
NAME
    main.py evaluate - Evaluate a model using the specified dataset and changing its parameters. We use k-fold cross validation repeated 5 times as the evaluation method. The results are saved in the results/<dataset>.csv

SYNOPSIS
    main.py evaluate DATASET <flags>

DESCRIPTION
    Evaluate a model using the specified dataset and changing its parameters. We use k-fold cross validation repeated 5 times as the evaluation method. The results are saved in the results/<dataset>.csv

POSITIONAL ARGUMENTS
    DATASET
        Type: str
        (str) Name of the dataset to be used.
        Possible values: ['iris', 'letter', 'mushroom', 'dis', 'shuttle', 'adult', 'breast_cancer', 'lupus', 'spambase']

FLAGS
    --seed=SEED
        Type: int
        Default: 1234
        (int, default 1234) Seed for random state.

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```
For example:
```{shell}
$ python main.py evaluate iris
```