import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# trying the folds on the KNN classifier:
from sklearn.neighbors import KNeighborsClassifier

from g6_helpers import accuracy_calcs_no_sklearn
from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean
from g6_helpers import standardize_k_folds, standardise_train_test_data

import sys


# running a knn model with 1 neighbor:


def knn(n_neighbors=1, standardise=True, half=False, kfold=False, per_class_scores=False):
    if n_neighbors is None:
        n_neighbors = 1
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    if kfold:

        # creating the k-folds with 10 folds:
        kfolds = get_k_folds(half=half)

        # standardizing the train and test data independently:
        if standardise:
            kfolds = standardize_k_folds(kfolds)

        # testing the model across the folds, and appending the result of each iteration to the knn_results list:
        scores = []
        for i, fold in enumerate(kfolds):
            knn.fit(fold[0], fold[1])
            y_pred = knn.predict(fold[2])
            fold_result = accuracy_calcs_no_sklearn(fold[3], y_pred, per_class=per_class_scores)
            scores.append(fold_result)
            # print("CV set " + str(i + 1) + " result: ", fold_result)
        means_of_scores = results_mean(scores)
        return means_of_scores, scores

    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)
        knn.fit(train_test_data[0], train_test_data[1])
        y_pred = knn.predict(train_test_data[2])
        score = accuracy_calcs_no_sklearn(train_test_data[3], y_pred, per_class=per_class_scores)
        return score, [score]


def test_knn():
    neighbors = range(1, 20)
    results = {}
    for k in neighbors:
        means_of_scores, scores = knn(n_neighbors=k, kfold=True)
        results[k] = means_of_scores
    accuracy_list = []
    for k in results:
        accuracy_list.append(results[k]['precision'])

    plt.plot(neighbors, accuracy_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('precision')

    plt.xticks(neighbors)
    plt.show()


if __name__ == '__main__':
    neighbors = range(1, 20)
    results = {}
    for k in neighbors:
        means_of_scores, scores = knn(n_neighbors=k, kfold=True)
        results[k] = means_of_scores
    accuracy_list = []
    for k in results:
        accuracy_list.append(results[k]['precision'])

    plt.plot(neighbors, accuracy_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('precision')
    plt.xticks(neighbors)
    plt.show()

