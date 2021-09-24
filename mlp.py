import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from g6_helpers import accuracy_calcs_no_sklearn
from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean, get_metric_from_scores
from g6_helpers import standardize_k_folds, standardise_train_test_data
from plotting import box_plot



def mlp(standardise=True, half=False, hidden_layer_sizes=(13, 13, 13), activation='tanh', rand_state=1, solver='lbfgs',

        max_iterations=200, kfold=False, per_class_scores=False):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=rand_state,
                        solver=solver, max_iter=max_iterations)

    if kfold:

        # creating the k-folds with 10 folds:
        kfolds = get_k_folds(half=half)

    # standardizing the train and test data independently:
        if standardise:
            kfolds = standardize_k_folds(kfolds)

        # testing the model across the folds, and appending the result of each iteration to the knn_results list:
        # rand_state = sys.argv[3] if len(sys.argv) > 3 else None

        # if len(sys.argv) == 4:
        #     try:
        #         rand_state = int(rand_state)
        #     except:
        #         print("Invalid input entered for random state")
        #         quit()
        #
        # if len(sys.argv) > 4:
        #     print("Too many parameters entered")
        #     quit()

        # elif rand_state is None:
        scores = []
        for i, fold in enumerate(kfolds):
            mlp.fit(fold[0], fold[1])
            y_pred = mlp.predict(fold[2])
            fold_result = accuracy_calcs_no_sklearn(fold[3], y_pred, per_class=per_class_scores)
            scores.append(fold_result)
        means_of_scores = results_mean(scores)
        return means_of_scores, scores

    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)
        mlp.fit(train_test_data[0], train_test_data[1])
        y_pred = mlp.predict(train_test_data[2])
        score = accuracy_calcs_no_sklearn(train_test_data[3], y_pred, per_class=per_class_scores)
        return score, [score]



    # elif rand_state < 1000:
    #     scores = []
    #     mlp = MLPClassifier(random_state=rand_state)
    #     for i, fold in enumerate(kfolds):
    #         mlp.fit(fold[0], fold[1])
    #         y_pred = mlp.predict(fold[2])
    #         fold_result = accuracy_calcs_no_sklearn(fold[3], y_pred, per_class=False)
    #         scores.append(fold_result)
    #     means_of_scores = results_mean(scores)
    #     return means_of_scores, scores
    #
    # else:
    #     print("Invalid parameter values")
    #     quit()


def test_mlp():
    activations = ['relu', 'tanh', 'logistic', 'identity']
    solvers = ['sgd', 'adam', 'lbfgs']
    titles = []
    perms = [[a, s] for a in activations for s in solvers]
    results = {}
    scorelist = []
    for i in perms:
        print(i)
        means_of_scores, scores = mlp(activation=i[0], solver=i[1], kfold=True)
        results[i[0] + ' & ' + i[1]] = means_of_scores
        titles.append(i[0] + ' & ' + i[1])
        scorelist.append(get_metric_from_scores(scores, 'precision'))

    print('scorelist: ', scorelist)
    print('perms: ', perms)

    box_plot(scorelist, 'precision', titles)

    plt.show()



if __name__ == '__main__':

    activations = ['relu', 'tanh', 'logistic', 'identity']
    solvers = ['sgd', 'adam', 'lbfgs']
    titles = []
    perms = [[a, s] for a in activations for s in solvers]
    results = {}
    scorelist = []
    for i in perms:
        print(i)
        means_of_scores, scores = mlp(activation=i[0], solver=i[1], kfold=True)
        results[i[0] + ' & ' + i[1]] = means_of_scores
        titles.append(i[0] + ' & ' + i[1])
        scorelist.append(get_metric_from_scores(scores, 'precision'))

    print('scorelist: ', scorelist)
    print('perms: ', perms)

    box_plot(scorelist, 'precision', titles)

    plt.show()



