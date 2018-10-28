# -*- coding: utf-8 -*-
import csv
import numpy as np
# Import all helper functions; packages listed explicitly for clarity.
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission


def sigmoid(t):
    """Apply sigmoid function elementwise to input scalar or vector."""
    return (1/(1+np.exp(-t)))

def build_poly(x, degree):
    """Build polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def standardize(x):
    """Standardize the original data set."""
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - np.mean(x[:,i])) / np.std(x[:,i])
        #x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())
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


def calculate_mse(e):
    """Calculate MSE value for an input error value or vector."""
    return (1/2)*np.mean(e**2)

def compute_loss(y, tx, w):
    """Computes loss for the error calculation function in the return statement."""
    e = y - tx.dot(w)
    return calculate_mse(e)

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
            # compute a stochastic gradient and loss
            err = y - tx.dot(w)
            grad = -tx.T.dot(err) / len(err)
            # update w through the stochastic gradient update
            w = w - gamma * grad
        # calculate loss
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
   aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
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
        grad = tx.T.dot( sigmoid(np.matmul(tx,w)) - y) + (lambda_*w)
        w = w - gamma * grad
    loss = grad
    return loss, w



"""
TRAINING
"""

y,x,i = load_csv_data('data/train.csv',sub_sample=False)

x = standardize(x)

# Add one dimenssion to X with only 1,beacause 1*W0+ x1*W1 + ...
b = np.ones((x.shape[0],1), dtype = int)
x = np.column_stack((b, x))

# Now creat vector Polynomial basis
#x = build_poly(x,31 )
initial_w = np.random.rand(x.shape[1],1)
y = y.reshape(y.shape[0],1)

"""Choose algorithm for training."""
MAX_ITERS = 100
GAMMA = 0.001
LAMBDA_ = 0.5
#loss, w = least_squares_GD(y,x,initial_w, MAX_ITERS, GAMMA)
#loss , w = least_squares_SGD(y, x, initial_w, MAX_ITERS, GAMMA, LAMBDA)
#loss, w = least_squares(y, x)
#loss, w = logistic_regression(y, x, initial_w, MAX_ITERS, GAMMA)
loss, w = reg_logistic_regression(y, x, initial_w, MAX_ITERS, GAMMA, LAMBDA_)

print(loss,w)

y_predictions = predict_labels(w, x)

# Generate test targets
y_test, x, i = load_csv_data('data/test.csv',sub_sample=False)

def calculate_prediction_accuracy(predictions, targets):
    """Calculate the prediction accuracy for predictions against targets."""
    correct = 0
    total_samples = len(predictions)
    for i in range(total_samples):
        if (targets[i] == y_predictions[i]):
            correct += 1
    return correct / total_samples

print("Prediction accurracy: ")
print(calculate_prediction_accuracy(y_predictions, y_test))
