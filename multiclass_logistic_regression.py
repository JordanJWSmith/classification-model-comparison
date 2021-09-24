import numpy as np
import pandas as pd
import scipy.stats as stats

from g6_helpers import accuracy_calcs_no_sklearn
from g6_helpers import kfold_train_test_split
from g6_helpers import standardize_k_folds, standardise_train_test_data

from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean

import sys


class MultiClassLR:
    """
       Trains weights given input data, targets and hyperparameters

       parameters
       ----------
       lr = learning rate
       n_iter = number of iterations

       returns
       -------
       weights - matrix of weights applied to feature values for calssification of input data
    """

    def fit(self, values_train, targets_train, lr=0.93, n_iter=14):

        self.unique_targets = np.unique(targets_train)
        # initialises empty weight matrix with the same dimensions as the input data
        self.weights = np.zeros((len(self.unique_targets), values_train.shape[1]))
        # turns target inputs into a 'one-hot vector'
        targets_train = self.ToOneHot(targets_train)
        # iterative decreasing error function by updating values of weight matrix
        for i in range(n_iter):
            z = np.dot(values_train, self.weights.T)
            h = self.softmax(z)
            gradient = np.dot(values_train.T, (h - targets_train)) / targets_train.size
            self.weights -= lr * gradient.T
        return self.weights

    # calculates likelihood of a data entries membership in a class
    def probabilities(self, x):
        scores = np.dot(x, self.weights.T)
        return self.softmax(scores)

    # generalised logistic function
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    # predicts most likely class given probability distribution
    def predict(self, x):
        return np.vectorize(lambda i: self.unique_targets[i])(np.argmax(self.probabilities(x), axis=1))

    # determines how close predicted values are to the target values
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

    def accuracy(self, targets_test, test_results, per_class_scores=False):
        accuracy_score = accuracy_calcs_no_sklearn(targets_test, test_results, per_class=per_class_scores)
        return accuracy_score

    # produces one-hot vector from input values
    def ToOneHot(self, y):
        num_classes = list(np.unique(y))
        one_hot = np.zeros((len(y), len(num_classes)))
        for i, c in enumerate(y):
            one_hot[i][num_classes.index(c)] = 1
        return one_hot


# runs MLR model with chosen hyperparameters
def run_mlr_model(lrx=0.93, n_iterx=14, half=False, kfold=False, per_class=False):
    mlr = MultiClassLR()

    if kfold:
        kfolds = get_k_folds(half=half)
        kfolds = standardize_k_folds(kfolds)
        scores = []
        for i, fold_split in enumerate(kfolds):
            mlr.fit(fold_split[0], fold_split[1], lr=lrx, n_iter=n_iterx)
            prediction = mlr.predict(fold_split[2])
            accuracy_score = mlr.accuracy(fold_split[3], prediction, per_class_scores=per_class)
            scores.append(accuracy_score)
            # print("CV set " + str(i+1) + " result: ", accuracy_score)
        means_of_scores = results_mean(scores)
        return means_of_scores, scores

    else:
        train_test_data = get_train_test_values_targets(half=half)
        train_test_data = standardise_train_test_data(train_test_data)
        mlr.fit(train_test_data[0], train_test_data[1], lr=lrx, n_iter=n_iterx)
        prediction = mlr.predict(train_test_data[2])
        score = mlr.accuracy(train_test_data[3], prediction, per_class_scores=per_class)
        return score, [score]


# function to test for optimal hyperparameters - values entered for learning rate are divided by 100000

def optimsation_test(lr_start=90000, lr_finish=100000, lr_interval=1000, itr_start=1, itr_finish=40, itr_interval=1):
    lr = range(lr_start, lr_finish, lr_interval)
    itr = range(itr_start, itr_finish, itr_interval)
    results = {}
    lrs = {}
    itrs = {}
    itr_list = []
    lr_list = []

    for i in itr:
        itr_list.append(i)
    for k in lr:
        lr_list.append(k)

    temp0 = []
    temp1 = []
    for x in itr_list:
        iterations_dict = {}
        for y in lr_list:
            iterations_dict.update([(x, y)])
            key, val = next(iter(iterations_dict.items()))
            temp0.append(key)
            temp1.append(val)
            lr_itr = zip(temp0, temp1)

    for i in lr_itr:
        means_of_scores, scores = run_mlr_model(lrx=(i[1] / 100000), n_iterx=(i[0]))
        results[i] = means_of_scores
        lrs[i] = i[1] / 100000
        itrs[i] = i[0]

    accuracy_list = []
    lr_list = []
    itrs_list = []
    for r in results:
        accuracy_list.append(results[r]['precision'])
        lr_list.append(lrs[r])
        itrs_list.append(itrs[r])

    max_precision = (max(accuracy_list))
    position_max = accuracy_list.index(max_precision)

    lr_optimal = lr_list[position_max]
    itrs_optimal = itrs_list[position_max]

    return print(f"Optimal precision: {max_precision}\nLearning rate: {lr_optimal}\nIterations: {itrs_optimal}")


if __name__ == '__main__':
    # mean, scrs = run_mlr_model()
    # print(mean)
    optimsation_test()
    import matplotlib.pyplot as plt

    plt.show()

# Optimal parameters for precisoon optimisation:
# half=False, kfold=False --> lr = 0.93, iter = 14, precision = 0.963
# half=True, kfold=False --> lr = 0.58, iter = 3, precision = 0.963
# half=False, kfold=True --> lr = 0.93, iter = 9, precision = 0.946
# half=True, kfold=True --> lr = 0.98, iter = 38, precision = 0.905
