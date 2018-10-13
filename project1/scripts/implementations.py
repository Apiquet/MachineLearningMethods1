# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
   
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
     return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

def 
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(err)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad     
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent.""" 
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            err = y - tx.dot(w)
            grad = -tx.T.dot(err) / len(err)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w


def ridge_regression(y, tx, lamb ):
    #L2 Regularization 
    N = len(y)
    lambprime = lamb * 2 * N
    w = np.invert(np.transpose(tx).dot(tx) + lambprime*np.identity(N)) * np.transpose(tx).dot(y)
    loss = compute_loss(y, tx, w)
    return loss,w


def logistic_regression(y, tx, initial w, max iters, gamma):
    """Code"""
    
    return y


def reg_logistic_regression(y, tx, lambda, initial w, max iters, gamma):
    """Code"""
    
    return y