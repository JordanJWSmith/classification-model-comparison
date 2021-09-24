import numpy as np
from fomlads import create_cv_folds
from fomlads import train_and_test_partition
from fomlads import standardise_data
import sys


# perform k folds
def perform_k_folds(dataframe, num_folds=10):
    n = dataframe.shape[0]
    return create_cv_folds(n, num_folds, rand_seed=7)


def split_targets_values(data):
    """

    Parameters
    ----------
    data - assuming dataframe given, with target as first column, remaining columns =values/inputs

    Returns
    -------
    numpy array
    """
    inputs = data.to_numpy()
    targets = inputs[:, 0]
    values = inputs[:, 1:]
    return targets, values


def check_fold_shape(input_array, target_array, folds):
    try:
        assert folds.shape[2] == len(input_array) == len(target_array)
    except AssertionError:
        print("Shape of fold does not match data. Exiting program...")
        sys.exit()


# train and test from folds
def kfold_train_test_split(dataframe, half, num_folds=10):
    t, v = split_targets_values(dataframe)
    folds = np.asarray(perform_k_folds(dataframe, num_folds))
    check_fold_shape(v, t, folds)

    folds_data = []
    for fold in folds:
        folds_data.append(train_and_test_partition(v, t, fold[0], fold[1], half))
    return np.asarray(folds_data)


# Accuracy calculations no sklean
def accuracy_calcs_no_sklearn(y_true, y_pred, per_class=False):
    """
    y_true - numpy array for true targets values
    y_pred - numpy array for predicted targets values
    per_class=true will return recall, precision and f1 separately for each class in arrays
    """
    # accuracy
    acc = ((y_true == y_pred).sum()) / y_true.shape[0]

    #     need multilabel confusion matrix

    pos = y_true == y_pred
    tp_vals = y_true[pos]
    wrong = y_true != y_pred
    wrong_vals_actual = y_true[wrong]
    wrong_vals_pred = y_pred[wrong]

    # calculate false positive and false negative for each class
    # get unique classes - make arrays to store scores
    classes = np.unique(y_true)
    per_class_rec = []
    per_class_prec = []
    per_class_f1 = []

    for cls in classes:
        #         print(cls)
        true_pos = tp_vals == cls
        # FP - predict this class, but belonged to different class
        false_pos = wrong_vals_pred == cls
        # FN is where we didn't predict this cls but correct was this class
        false_neg = wrong_vals_actual == cls

        recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())
        precision = true_pos.sum() / (true_pos.sum() + false_pos.sum())
        f1 = (2 * precision * recall) / (precision + recall)
        per_class_rec.append(recall)
        per_class_prec.append(precision)
        per_class_f1.append(f1)

    if per_class:
        return {"accuracy": acc, "precision": per_class_prec, "recall": per_class_rec, "f1": per_class_f1}

    else:

        # macro avg scores
        return {"accuracy": acc, "precision": np.average(per_class_prec), "recall": np.average(per_class_rec),
                "f1": np.average(per_class_f1)}


def standardize_k_folds(all_folds):
    """
       Standardise the k-folds train inputs and test inputs independently of one another

       parameters
       ----------
       the k-folds data matrix

       returns
       -------
       std_datamtx - data matrix where columns have mean 0 and variance 1
       """
    standardized_folds = []
    for train_inputs, train_targets, test_inputs, test_targets in all_folds:
        train_inputs = standardise_data(train_inputs)
        test_inputs = standardise_data(test_inputs)
        standardized_folds.append([train_inputs, train_targets, test_inputs, test_targets])

    return standardized_folds


def standardise_train_test_data(train_test_data):
    train_v = standardise_data(train_test_data[0])
    train_t = train_test_data[1]
    test_v = standardise_data(train_test_data[2])
    test_t = train_test_data[3]
    return np.array([train_v, train_t, test_v, test_t])


def initial_train_test_split(df):
    np.random.seed(7)
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train, test


if __name__ == '__main__':
    from get_data import get_data

    train, test = initial_train_test_split(get_data('wine_data.csv'))
    print(train)
    print(test)
