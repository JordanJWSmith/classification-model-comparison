import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB

from g6_helpers import accuracy_calcs_no_sklearn
from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean
from g6_helpers import standardize_k_folds, standardise_train_test_data


# Run naive Bayes classifier
def naive_bayes(standardise=False, half=False, kfold=False, per_class_scores=False):
    gnb = GaussianNB()

    if kfold:

        # creating the k-folds with 10 folds:
        kfolds = get_k_folds(half=half)

        # standardizing the train and test data independently:
        if standardise:
            kfolds = standardize_k_folds(kfolds)

        # testing the model across the folds, and appending the result of each iteration to the scores list:
        scores = []
        for i, fold in enumerate(kfolds):
            gnb.fit(fold[0], fold[1])
            y_pred = gnb.predict(fold[2])
            fold_result = accuracy_calcs_no_sklearn(fold[3], y_pred, per_class=per_class_scores)
            scores.append(fold_result)

        means_of_scores = results_mean(scores)
        return means_of_scores, scores

    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)
        gnb.fit(train_test_data[0], train_test_data[1])
        y_pred = gnb.predict(train_test_data[2])
        score = accuracy_calcs_no_sklearn(train_test_data[3], y_pred, per_class=per_class_scores)
        return score, [score]


if __name__ == '__main__':
    mean, scrs = naive_bayes()
    print(mean)
    import matplotlib.pyplot as plt

    plt.show()
