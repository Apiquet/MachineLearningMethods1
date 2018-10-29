# -*- coding: utf-8 -*-

"""
@author: SMARDA/ERBACHER/PIQUET
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
# Import all helper functions; packages listed explicitly for clarity.
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission


"""
MATHEMATICAL HELPER FUNCTIONS
"""
def sigmoid(t):
    """Apply sigmoid function elementwise to input scalar or vector."""
    return (1/(1+np.exp(-t)))

def standardize(x):
    """Standardize the original data set."""
    for col_idx in range(x.shape[1]):
        x[:,col_idx] = (x[:,col_idx] - np.mean(x[:,col_idx])) / np.std(x[:,col_idx])
    return x

def calculate_mse(e):
    """Calculate MSE value for an input error value or vector."""
    return (1/2)*np.mean(e**2)

def compute_loss(y, tx, w):
    """Computes loss for the error calculation function in the
    return statement."""
    e = y - tx.dot(w)
    return calculate_mse(e)


"""
PREPROCCESSING FUNCTIONS
"""
def remove_columns(mtx):
    """Removes columns from the dataset where too many values are -999."""
    columns_to_remove = []
    COLUMN_REMOVE_THRESHOLD = -250
    for col_idx in range(mtx.shape[1]):
        #print(np.sum(x[:,i])/x.shape[0])
        if np.sum(mtx[:,col_idx])/mtx.shape[0] < COLUMN_REMOVE_THRESHOLD:
            columns_to_remove.append(col_idx)
    mtx_cols_removed = np.delete(mtx, columns_to_remove, axis=1)
    return mtx_cols_removed

def remove_rows(mtx, labels):
    """
    Remove rows from the dataset and labels where any value is -999.

    In it's current implementation this function requires substantial amounts
    of memory. For this reason this function is currently never called. If
    a more efficient implementation is created, this function could be called
    directly after the concatenation step in the crossvalidation() fuction.
    """

    rows_with_bad_data = np.where(mtx == -999)[0]
    # Remove duplicates for rows with multiple values of -999
    rows_to_remove = np.unique(rows_with_bad_data)
    mtx_rows_removed = np.delete(mtx, rows_to_remove, axis=0)
    y_rows_removed = np.delete(labels, rows_to_remove)
    return mtx_rows_removed, y_rows_removed

def replace_outliers_with_mean(mtx):
    """Replace values of -999 with the mean of the column."""
    for col_idx in range(mtx.shape[1]):
        col_for_calculating_mean = mtx[:,col_idx]
        cells_with_bad_data = np.where(mtx[:,col_idx] == -999)
        column_bad_cells_removed = np.delete(col_for_calculating_mean, cells_with_bad_data)
        column_mean = np.mean(column_bad_cells_removed)

        column_for_replacement = mtx[:,col_idx]
        column_for_replacement[column_for_replacement == -999] = column_mean
        mtx[:,col_idx] = column_for_replacement
    return mtx

def build_poly(x, degree):
    """Build polynomial basis functions for input data x, for j=0 up to
    j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def shuffle_data(tx, y):
      data_size = len(y)
      # Seed random number generator for consistent results
      np.random.seed(1)
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_y = y[shuffle_indices]
      shuffled_tx = tx[shuffle_indices]
      return shuffled_tx, shuffled_y

def addones(x):
    """Add one dimenssion to x with only 1, for W0."""
    b = np.ones((x.shape[0],1), dtype = int)
    x = np.column_stack((b, x))#
    return x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Create batches for batch gradient descent algorithm."""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


"""
TRAINING ALGORITHMS
"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Execute gradient descent algorithm."""
    w = initial_w[:]
    for i in range(max_iters):
        e = y - tx.dot(w)
        grad = - tx.T.dot(e) / len(e)
        w = w - gamma * grad
    loss = calculate_mse(e)
    return loss, w

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Execute stochastic gradient descent algorithm."""
    w = initial_w[:]
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # Compute a stochastic gradient and loss
            err = y_batch - tx_batch.dot(w)
            grad = -tx_batch.T.dot(err) / len(err)
            w = w - gamma * grad
        loss = compute_loss(y, tx, w)
    return loss, w

def least_squares(y, tx):
    """Calculate weights using least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss(y, tx, w)
    return loss, w

def ridge_regression(y, tx, lambda_ ):
   """Calculate weights using ridge regression."""
   aI = lambda_ * np.identity(tx.shape[1])
   a = tx.T.dot(tx) + aI
   b = tx.T.dot(y)
   return np.linalg.solve(a, b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Calculate weights using logistic regression."""
    w = initial_w[:]
    for i in range(max_iters):
        grad = np.matmul(tx.T, sigmoid(np.matmul(tx,w)) - y)
        w = w - gamma * grad
    loss = grad
    return loss, w

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """Calculate weights using regularized logistic regression."""
    w = initial_w[:]
    for i in range(max_iters):
        grad = np.matmul(tx.T, sigmoid(np.matmul(tx,w)) - y) + (lambda_*w)
        w = w - gamma * grad
    loss = grad
    return loss, w

def train(y, x, param):
    """Choose algorithm for training.

    Positional parameters:
    y ------ data labels.
    x ------ input data.
    param -- List of length 4 with following index/value map:
             0 -> maximum iterations for optimization algorithm.
             1 -> Optimization rate parameter for relevant algorithms.
             2 -> Regularization parameter for relevant algorithms.
             3 -> Specification for which training algorithm to use.
    """
    MAX_ITERS = param[0]
    GAMMA = param[1]
    LAMBDA_ = param[2]
    f = param[3]
    initial_w = np.random.rand(x.shape[1],1)
    if f == 0:
        loss, w = least_squares_GD(y,x,initial_w, MAX_ITERS, GAMMA)
    elif f == 1:
        loss , w = least_squares_SGD(y, x, initial_w, 1, MAX_ITERS, GAMMA)
    elif f == 2:
        loss,w = ridge_regression(y, x, LAMBDA_)
    elif f == 3:
        loss, w = least_squares(y, x)
    elif f == 4:
        loss, w = logistic_regression(y, x, initial_w, MAX_ITERS, GAMMA)
    elif f == 5:
        loss, w = reg_logistic_regression(y, x, initial_w, MAX_ITERS, GAMMA, LAMBDA_)
    return w


"""
PROCESSING & POSTPROCESSING ALGORITHMS
"""
def calculate_prediction_accuracy(y_predictions, targets):
    """Calculate the prediction accuracy for predictions against targets."""
    correct = 0
    total_samples = len(y_predictions)
    for i in range(total_samples):
        if (targets[i] == y_predictions[i]):
            correct += 1
    return correct / total_samples

def crossvalidation(y,x,k,n,param):
    #data divided in n part, validate on subset k and train on the other n-k"
    """
    Trains and evaluates model for given K-fold CV parameter set.

    For an input matrix x and total of n folds, this function trains and
    evaluates the model. The intention of this function is for it to be
    called n times. Each time, the test set is composed of the rows from the
    training data from row N to row N=k to row N=k+N/n. For example, if the
    dataset has a total of 1000 rows, and this function is called with arguments
    k=3, n=10, this function will use x[300:400] (with y[300:400] as the
    corresponding labels) as the test dataset and all remaining datapoints
    for training (i.e., concatenation of x[0:300] and x[400:1000]).

    Positional parameters:
    y ------ All labels available for training.
    x ------ Input training data matrix; should be shuffled prior to passing
             to the cross_validation function.
    k ------ Which row to begin data segmentation at.
    n ------ Total number of folds in cross-validation.
    param -- list of parameters used in train() function. See implementation
             of train() for more details.
    """
    x_validate = x[k:k + x.shape[0] // n]
    y_validate = y[k:k + y.shape[0] // n]

    x_train = np.concatenate((x[:k], x[k:k+x.shape[0]+1]), axis = 0)
    y_train = np.concatenate((y[:k], y[k:k+y.shape[0]+1]), axis = 0)

    x_validate = replace_outliers_with_mean(x_validate)
    x_train = replace_outliers_with_mean(x_train)

    x_train = standardize(x_train)
    x_validate = standardize(x_validate)

    x_train = addones(x_train)
    x_validate = addones(x_validate)

    w = train(y_train, x_train, param)
    y_predictions = predict_labels(w, x_validate)
    accuracy = calculate_prediction_accuracy(y_predictions, y_validate)
    return accuracy, y_predictions, w

def plot_result(lambdas, accuracies):
    plt.plot(lambdas, accuracies)
    plt.title('accuracy(Lambda)')
    plt.xlabel('Lambda(log)')
    plt.ylabel('Accuracy')
    plt.xscale('log')

def predict_reverse(y_pred):
    """Generate a proper Y to submission with -1"""
    y_pred[np.where(y_pred == 0)] = -1
    return y_pred

def submission(x_test,w,i):
    x_test = remove_columns(x_test)
    x_test = replace_outliers_with_mean(x_test)
    x_test = standardize(x_test)
    x_test = addones(x_test)
    y_predictions = predict_labels(w, x_test)
    y_predictions = predict_reverse(y_predictions)
    y_predictions.reshape(y_predictions.shape[0],)
    create_csv_submission(i, y_predictions, 'data/sample-submission.csv')


"""
MAIN
"""
def main(param):
    # load train set
    y, x, i = load_csv_data('data/train.csv', sub_sample=False)
    # load test set
    y_test, x_test, i_test = load_csv_data('data/test.csv', sub_sample=False)
    # Reshape y
    y = y.reshape(y.shape[0], 1)
    # Preprocess x (remove features with lot of -999)
    x = remove_columns(x)

    # Number of sub-set for crossvalidation
    N_TOTAL_FOLDS = 1
    accuracies= []
    x, y = shuffle_data(x, y)
    for k in range(0, x.shape[0], x.shape[0] // N_TOTAL_FOLDS):
        accuracy, y_predictions, w = crossvalidation(y, x, k, N_TOTAL_FOLDS, param)
        accuracies.append(accuracy)
        print(accuracies)
        
    #plot_result(lambdas,accuracies)
    submission(x_test, w, i_test)
