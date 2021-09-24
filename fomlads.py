import numpy as np


def create_cv_folds(N, num_folds, rand_seed):
    """
    Defines the cross-validation splits for N data-points into num_folds folds.
    Returns a list of folds, where each fold is a train-test split of the data.
    Achieves this by partitioning the data into num_folds (almost) equal
    subsets, where in the ith fold, the ith subset will be assigned to testing,
    with the remaining subsets assigned to training.
    parameters
    ----------
    N - the number of datapoints
    num_folds - the number of folds
    returns
    -------
    folds - a sequence of num_folds folds, each fold is a train and test array
        indicating (with a boolean array) whether a datapoint belongs to the
        training or testing part of the fold.
        Each fold is a (train_part, test_part) pair where:
        train_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the training set, and False if
            otherwise.
        test_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the testing set, and False if
            otherwise.
    """
    # if the number of datapoints is not divisible by folds then some parts
    # will be larger than others (by 1 data-point). min_part is the smallest
    # size of a part (uses integer division operator //)
    min_part = N // num_folds
    # rem is the number of parts that will be 1 larger
    rem = N % num_folds
    # create an empty array which will specify which part a datapoint belongs to
    parts = np.empty(N, dtype=int)
    start = 0
    for part_id in range(num_folds):
        # calculate size of the part
        n_part = min_part
        if part_id < rem:
            n_part += 1
        # now assign the part id to a block of the parts array
        parts[start:start + n_part] = part_id * np.ones(n_part)
        start += n_part
    # now randomly reorder the parts array (so that each datapoint is assigned
    # a random part.
    np.random.seed(rand_seed)  # setting random seed value for reproducible results
    np.random.shuffle(parts)
    # we now want to turn the parts array, into a sequence of train-test folds
    folds = []
    for f in range(num_folds):
        train = (parts != f)
        test = (parts == f)
        folds.append((train, test))

    return folds


def train_and_test_partition(inputs, targets, train_filter, test_filter, half):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_filter - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_filter - (like train_filter) but specifying the test points.

    returns
    -------
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    # get the indices of the train and test portion
    if len(inputs.shape) == 1:
        # if inputs is a sequence of scalars we should reshape into a matrix
        inputs = inputs.reshape((inputs.size, 1))

    train_inputs = inputs[train_filter, :]
    test_inputs = inputs[test_filter, :]
    train_targets = targets[train_filter]
    test_targets = targets[test_filter]
    if half:
        idx = np.random.choice(train_inputs.shape[0], train_inputs.shape[0] // 2, replace=False)
        train_inputs = train_inputs[idx]
        train_targets = train_targets[idx]
    return train_inputs, train_targets, test_inputs, test_targets


def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix
    """
    # the mean sample is the mean of the rows of data
    N, dim = data.shape
    mu = np.mean(data, 0)
    Sigma = np.zeros((dim, dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T
        x_minus_mu = np.matrix(x - mu).reshape((dim, 1))
        # the outer-product v * v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma, dtype="object")


# Function to standardise data (taken from fomlads)

def standardise_data(datamtx):
    """
    Standardise a data-matrix

    parameters
    ----------
    datamtx - (NxD) data matrix (array-like)

    returns
    -------
    std_datamtx - data matrix where columns have mean 0 and variance 1
    """
    means = np.mean(datamtx, axis=0)
    stds = np.std(datamtx, axis=0)
    return (datamtx - means) / stds


