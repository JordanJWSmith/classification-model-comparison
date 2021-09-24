import pandas as pd
import numpy as np
import random
from dabl import plot
import matplotlib.pyplot as plt

random.seed(7)

from g6_helpers import initial_train_test_split
from g6_helpers import kfold_train_test_split, split_targets_values

data_file = 'wine_data.csv'


# retrieves the data and converts it into a pandas dataframe:
def get_data(csv):
    # Cleaned data still needed
    # Get CSV data and apply column names
    headers = ['Type', 'Alcohol', 'Malic', 'Ash',
               'Alcalinity', 'Magnesium', 'Phenols',
               'Flavanoids', 'Nonflavanoids',
               'Proanthocyanins', 'Color', 'Hue',
               'Dilution', 'Proline']
    return pd.read_csv(csv, names=headers)


# uses the kfold_train_test_split helper function to return an array that contains the train data split into folds:
def get_k_folds(num_folds=10, half=False):
    df = get_data(data_file)
    train, test = initial_train_test_split(df)
    kfolds = kfold_train_test_split(train, half, num_folds)
    return kfolds


def get_train_test_values_targets(half=False):
    df = get_data(data_file)
    train_df, test_df = initial_train_test_split(df)
    if half:
        train_df = train_df.sample(frac=0.5, replace=False)

    t_train, v_train = split_targets_values(train_df)
    t_test, v_test = split_targets_values(test_df)
    return np.array([v_train, t_train, v_test, t_test])


# returns the initial test dataset as a dataframe:
def get_initial_test():
    df = get_data(data_file)
    train, test = initial_train_test_split(df)
    return test

def plot_histogram():
    df = get_data(data_file)
    plot(df, 'Type')
    plt.show()


if __name__ == '__main__':
    test = get_initial_test()
    print(test)
    plot_histogram()