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
- cv
- ...

### How to check the method usage
To check how to use a specific method, you can run:
```{shell}
$ python main.py <method> -h
```
This will show you the mandatory and optional parameters
as well a description of the method. For example:
```{shell}
$ python main.py cv -h
```
```
NAME
    main.py cv - Cross-validate a model.

SYNOPSIS
    main.py cv DATASET K <flags>

DESCRIPTION
    Cross-validate a model.

POSITIONAL ARGUMENTS
    DATASET
        Type: str
        (str) Path to the dataset.
    K
        Type: int
        (int) Number of neighbors to consider on classification. Must be greater than 1.

FLAGS
    --dist=DIST
        Type: str
        Default: 'euclidean'
        (str, default 'euclidean') Distance metric. Possible values: euclidean, (TBD)...

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```
