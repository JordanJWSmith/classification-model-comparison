from sklearn.ensemble import RandomForestClassifier

from get_data import get_k_folds, get_train_test_values_targets
from calculate_result import results_mean, get_metric_from_scores
from plotting import box_plot
from g6_helpers import accuracy_calcs_no_sklearn, standardize_k_folds, standardise_train_test_data


def train_test_rfc(x_train, y_train, x_test, y_test, estimators_no=33, rand_state=342, per_class_scores=False,
                   max_feat="auto"):
    rf = RandomForestClassifier(n_estimators=estimators_no, random_state=rand_state, max_features=max_feat)
    rf.fit(x_train, y_train)
    predicted = rf.predict(x_test)
    score = accuracy_calcs_no_sklearn(y_test, predicted, per_class=per_class_scores)
    return score


def run_rfc(num_estimators=100, standardise=False, half=False, kfold=False, per_class=False, max_features="auto"):
    if kfold:
        kfolds = get_k_folds(half=half)
        if standardise:
            kfolds = standardize_k_folds(kfolds)
        scores = []
        for i, fold_split in enumerate(kfolds):
            result = train_test_rfc(fold_split[0], fold_split[1], fold_split[2], fold_split[3],
                                    estimators_no=num_estimators, max_feat=max_features, per_class_scores=per_class)
            scores.append(result)

        means_of_scores = results_mean(scores)
        return means_of_scores, scores
    else:
        train_test_data = get_train_test_values_targets(half=half)
        if standardise:
            train_test_data = standardise_train_test_data(train_test_data)

        score = train_test_rfc(train_test_data[0], train_test_data[1], train_test_data[2],
                               train_test_data[3], estimators_no=num_estimators, per_class_scores=per_class)
        return score, [score]


def run_test_params_rfc(standardise=False, acc_param="precision", max_features="auto", per_class=False):
    kfolds = get_k_folds()
    if standardise:
        kfolds = standardize_k_folds(kfolds)
    print()
    test_params = [100 // (x + 1) for x in range(len(kfolds))]
    print(test_params)
    scores = []
    for x in range(len(test_params)):
        param_score = []
        for i, fold_split in enumerate(kfolds):
            result = train_test_rfc(fold_split[0], fold_split[1], fold_split[2], fold_split[3],
                                    estimators_no=test_params[x], max_feat=max_features, per_class_scores=per_class)
            param_score.append(result)
        if per_class:
            c1, c2, c3 = get_metric_from_scores(param_score, acc_param, per_class=per_class)
            scores.append(c1)
            scores.append(c2)
            scores.append(c3)
        else:
            scores.append(get_metric_from_scores(param_score, acc_param, per_class=per_class))
        # scores.append(param_score)
    print(scores)
    if per_class:
        if standardise:
            box_plot(scores,
                     acc_param.capitalize() + ' per class - RFC num estimators (Standarised data) ' + "max_feat=" + max_features,
                     [str(x) for x in test_params])
        else:
            box_plot(scores, acc_param.capitalize() + ' per class - RFC num estimators ' + "max_feat=" + max_features,
                     [str(x) + " C" + str(y+1) for x in test_params for y in range(3)])
    else:

        if standardise:
            box_plot(scores,
                     acc_param.capitalize() + ' RFC num estimators (Standarised data) ' + "max_feat=" + max_features,
                     [str(x) for x in test_params])
        else:
            box_plot(scores, acc_param.capitalize() + ' RFC num estimators ' + "max_feat=" + max_features,
                     [str(x) for x in test_params])
            # [str(x) + " " + str(y) for x in test_params for y in range(3)])


if __name__ == '__main__':
    run_test_params_rfc(per_class=False)
    import matplotlib.pyplot as plt

    plt.show()
