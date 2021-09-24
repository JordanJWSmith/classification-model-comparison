import numpy as np
import pandas as pd
import scipy.stats as stats

from fomlads import max_lik_mv_gaussian
from g6_helpers import accuracy_calcs_no_sklearn
from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean, get_metric_from_scores
from plotting import box_plot
from g6_helpers import standardize_k_folds, standardise_train_test_data


def train_test_shared_covariance_model(values_train, targets_train, values_test, targets_test, per_class_scores=False):
    unique_targets = np.unique(targets_train)

    inputs0 = values_train[targets_train == 1, :]
    inputs1 = values_train[targets_train == 2, :]
    inputs2 = values_train[targets_train == 3, :]

    # get dimensions of training set
    n, d = values_train.shape

    # Get shapes of classes
    n0 = inputs0.shape[0]
    n1 = inputs1.shape[0]
    n2 = inputs2.shape[0]

    # Calculate priors for classes
    pi0 = n0 / n
    pi1 = n1 / n
    pi2 = n2 / n

    mean0, S0 = max_lik_mv_gaussian(inputs0)
    mean1, S1 = max_lik_mv_gaussian(inputs1)
    mean2, S2 = max_lik_mv_gaussian(inputs2)
    #  now we have pi, mean for each class and covmtx
    covmtx = (n0 / n) * S0 + (n1 / n) * S1 + (n2 / n) * S2

    # predict for test

    # Calculate class densities for each data point (p(xn|Ck))
    class0_densities = stats.multivariate_normal.pdf(values_test, mean0, covmtx)
    class1_densities = stats.multivariate_normal.pdf(values_test, mean1, covmtx)
    class2_densities = stats.multivariate_normal.pdf(values_test, mean2, covmtx)

    # Evaluate posterior class probability for every data point (p(Ck|xn))
    posterior_probs0 = (pi0 * class0_densities) / (
            (pi1 * class1_densities) + (pi0 * class0_densities) + (pi2 * class2_densities))
    posterior_probs1 = (pi1 * class1_densities) / (
            (pi1 * class1_densities) + (pi0 * class0_densities) + (pi2 * class2_densities))
    posterior_probs2 = (pi2 * class2_densities) / (
            (pi1 * class1_densities) + (pi0 * class0_densities) + (pi2 * class2_densities))

    # combine probabilities of each class into one array - return probs + final prediction (i.e. max likelihood)
    probabilities = []
    for i in range(len(values_test)):
        probabilities.append({
            1: posterior_probs0[i],
            2: posterior_probs1[i],
            3: posterior_probs2[i],
        })
    predicted = [max(probabilities[i].keys(), key=(lambda key: probabilities[i][key])) for i in
                 range(len(probabilities))]
    probabilities, predicted = np.asarray(probabilities), np.asarray(predicted)

    # Now we have max likelihood, calculate accuracy score.
    score = accuracy_calcs_no_sklearn(targets_test, predicted, per_class=per_class_scores)
    return score


def run_covariance_model(standardise=False, half=False, kfold=False, per_class=False):
    # if kfold, run 10 fold cross validation + return the mean OR SCORES AND SCORES
    if kfold:
        kfolds = get_k_folds(half=half)
        if standardise:
            kfolds = standardize_k_folds(kfolds)
        scores = []
        for i, fold_split in enumerate(kfolds):
            result = train_test_shared_covariance_model(fold_split[0], fold_split[1], fold_split[2], fold_split[3],
                                                        per_class_scores=per_class)
            scores.append(result)

        # return scores
        means_of_scores = results_mean(scores)
        return means_of_scores, scores
    # if no kfold, run params on normal train test split
    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)

        score = train_test_shared_covariance_model(train_test_data[0], train_test_data[1], train_test_data[2],
                                                   train_test_data[3], per_class_scores=per_class)
        return score, [score]


if __name__ == '__main__':
    mean, scrs = run_covariance_model()
    print(mean)
    import matplotlib.pyplot as plt

    plt.show()
