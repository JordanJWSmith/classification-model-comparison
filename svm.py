from sklearn.svm import LinearSVC  # Linear Support Vector Classification.

from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean, get_metric_from_scores
from plotting import box_plot
from g6_helpers import accuracy_calcs_no_sklearn, standardize_k_folds, standardise_train_test_data
import matplotlib.pyplot as plt


def train_test_svm(x_train, y_train, x_test, y_test, per_class_scores=False, rand_state=87, max_iters=3000,
                   c_value=1.0, loss_f="squared_hinge"):
    # multiclass linear, using one-vs-rest strategy
    lsvc = LinearSVC(random_state=rand_state, max_iter=max_iters, C=c_value, loss=loss_f).fit(x_train,
                                                                                              y_train)  # random state for reproducible results.
    predicted = lsvc.predict(x_test)
    score = accuracy_calcs_no_sklearn(y_test, predicted, per_class=per_class_scores)
    return score


def run_svm(standardise=True, half=False, rand_state=87, max_iters=3000, kfold=False, c_val=1.0,
            loss_func="squared_hinge", per_class=False):
    if kfold:
        kfolds = get_k_folds(half=half)
        if standardise:
            kfolds = standardize_k_folds(kfolds)  # svm works better with standardised.
        scores = []
        for i, fold_split in enumerate(kfolds):
            result = train_test_svm(fold_split[0], fold_split[1], fold_split[2], fold_split[3], rand_state=rand_state,
                                    max_iters=max_iters, c_value=c_val, loss_f=loss_func, per_class_scores=per_class)
            scores.append(result)

        # print(scores)
        means_of_scores = results_mean(scores)
        return means_of_scores, scores
    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)

        score = train_test_svm(train_test_data[0], train_test_data[1], train_test_data[2], train_test_data[3],
                               rand_state=rand_state, max_iters=max_iters, per_class_scores=per_class)
        return score, [score]


def test_svm(standardise=True, half=False, rand_state=87, max_iters=3000, acc_param="precision", loss="squared_hinge"):
    kfolds = get_k_folds(half=half)
    if standardise:
        kfolds = standardize_k_folds(kfolds)

    test_param_c = [0.1, 1.0, 1.5, 3, 5, 10, 100]
    scores = []
    for x in range(len(test_param_c)):
        param_score = []
        for i, fold_split in enumerate(kfolds):
            result = train_test_svm(fold_split[0], fold_split[1], fold_split[2], fold_split[3], rand_state=rand_state,
                                    max_iters=max_iters, c_value=test_param_c[x], loss_f=loss)
            param_score.append(result)
        scores.append(get_metric_from_scores(param_score, acc_param))
    print(scores)
    if standardise:
        box_plot(scores, acc_param.capitalize() + ' LinearSVC (Standarised data)',
                 [str(x) for x in test_param_c])
    else:
        box_plot(scores, acc_param.capitalize() + ' LinearSVC (non-standarised data)',
                 [str(x) for x in test_param_c])


if __name__ == '__main__':
    # svm_mean, svm_score = run_svm()
    # print(svm_mean)
    # svm_acc = [get_metric_from_scores(svm_score, 'precision')]
    # box_plot(svm_acc, "Linear SVC Accuracy", ['LinearSCV'])
    test_svm(standardise=True, half=False, acc_param="precision", loss="squared_hinge")
    plt.show()
