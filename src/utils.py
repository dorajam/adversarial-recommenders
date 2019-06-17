from copy import copy
from scipy import io
from scipy.spatial import distance
import scipy.stats as st
import tensorflow as tf
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def generate_k_fold_split_datasets(original, k):
    k_trainx, k_testx = [],[]
    for i in range(k):
        train_x, test_x = get_train_and_test(original, seed=i)
        k_trainx.append(train_x)
        k_testx.append(test_x)
    return k_trainx, k_testx


def _create_aux_row(ls_of_indeces, items):
    row = np.zeros((1,items))
    for idx in ls_of_indeces:
        row[0][idx] = 1
    return row


def create_aux_matrix(subsets,items):
    """Creates an auxiliary matrix from a list of subsets.
    Subsets represents items that should be recommended together."""
    return np.vstack([_create_aux_row(row, items) for row in subsets])


def get_mean_with_ci(arr, confidence=0.95):
    mean = np.mean(arr)
    ci = st.t.interval(confidence, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
    return mean, ci


def precision(purchased, recommended, k):
    return len(np.intersect1d(purchased, recommended[:k])) / float(k)


def get_precision(pred_y, train_x, test_x, k=1, verbose=False):
    """pred_y is real valued, train_x and test_x are binary vectors."""
    precisions = []
    for user in xrange(train_x.shape[0]):
        train_x_ui = train_x[user]
        pred_y_ui = pred_y[user]
        test_x_ui = test_x[user]
        training_columns = np.nonzero(pred_y_ui * (train_x_ui != 0))[0]
        sorted_scores = np.argsort(pred_y_ui * (train_x_ui == 0))[::-1]
        recommended_columns = filter(lambda col_index: col_index not in training_columns, sorted_scores)
        test_columns = np.nonzero(test_x_ui)[0]

        if verbose:
            print '----------------------------'
            print 'User', user
            print 'Training data', train_x_ui
            print 'Prediction scores', pred_y_ui
            print 'Test data', test_x_ui
            print 'Training columns:', training_columns
            print 'Recommended columns:', recommended_columns
            print 'Test columns:', test_columns

        precisions.append(precision(test_columns, recommended_columns, k=k))
        if verbose:
            print 'Precision:', precisions[-1]
    return np.mean(precisions)


def get_aux_precision(pred_y, train_x, auxiliary_x, k=1, verbose=False):
    precisions = []
    for ui in xrange(train_x.shape[0]):
        train_x_ui = train_x[ui]
        pred_y_ui = pred_y[ui]
        training_columns = np.nonzero(pred_y_ui * (train_x_ui != 0))[0]
        sorted_scores = np.argsort(pred_y_ui * (train_x_ui == 0))[::-1]
        recommended_columns = filter(lambda col_index: col_index not in training_columns, sorted_scores)

        if verbose:
            print '----------------------------'
            print 'User', ui
            print 'Training data', train_x_ui
            print 'Prediction scores', pred_y_ui
            print 'Training columns:', training_columns
            print 'Recommended columns:', recommended_columns

        for auxi in xrange(auxiliary_x.shape[0]):
            aux_columns = np.nonzero(auxiliary_x[auxi])[0]
            if verbose:
                print 'Auxiliary data', aux_columns
            precisions.append(precision(aux_columns, recommended_columns, k=k))
            if verbose:
                print 'Precision', precisions[-1]
    return np.mean(precisions)


def apply_noise_if_specified(x, noise_parameter=None, seed=42):
    """Applies Gaussian noise to x with stddev equal to the inputted noise parameter"""
    if noise_parameter:
        noise = tf.truncated_normal(tf.shape(x), stddev=noise_parameter, seed=seed)
        return x + noise
    return x


def set_to_zero(row, seed=42):
    zero_idx = np.nonzero(row)[0]
    if any(zero_idx):
        np.random.seed(seed)
        row[np.random.choice(zero_idx)] = 0
    return row


def get_train_and_test(train_x, seed=42):
    copied_train_x = copy(train_x)
    np.apply_along_axis(set_to_zero, 1, copied_train_x, seed=seed)
    test_x = train_x - copied_train_x
    return copied_train_x, test_x


def set_top_prediction_to_one(row):
    ranked = len(row) + 1 - st.rankdata(row, 'ordinal').astype(int)
    non_one_indeces = np.where(ranked != 1)
    ranked[non_one_indeces] = 0
    return ranked


def get_final_top_predictions(predictions, train_x):
    predictions = predictions * (1 - train_x)
    return np.apply_along_axis(set_top_prediction_to_one, 1, predictions)


def get_precision_at_1(predictions, test_x):
    return np.sum(np.multiply(predictions, test_x)) / float(predictions.shape[0])


def get_mean_with_ci(arr, confidence=0.95):
    mean = np.mean(arr)
    ci = st.t.interval(confidence, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
    return mean, ci


def shuffle_users(x, seed=42):
    np.random.seed(seed)
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    return x[perm]
