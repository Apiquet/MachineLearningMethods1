# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
# Import all packages; packages listed explicitly for clarity.
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission



def sigmoid(t):
     return (1/(1+np.exp(-t)))
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

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
     return (1/2)*np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)


    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
   
    w = initial_w[:]
    for i in range(max_iters):
        e = y - tx.dot(w)
        grad = - tx.T.dot(e) / len(e)
        w = w - gamma * grad
    loss = calculate_mse(e)
    return loss, w


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent.""" 
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
   
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = compute_loss(y, tx, w)
    return loss, w


def ridge_regression(y, tx, lambda_ ):
   """implement ridge regression."""
   aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
   a = tx.T.dot(tx) + aI
   b = tx.T.dot(y)
   return np.linalg.solve(a, b)


def logistic_regression(y,tx,initial_w,max_iters,gamma):
    w = initial_w[:]
    for i in range(max_iters):
        grad = np.matmul(tx.T, sigmoid(tx.dot(w) - y))
        w = w - gamma * grad
    loss = grad
    return loss, w

def reg_logistic_regression(y,tx,initial_w,max_iters,gamma,lambda_):
    w = initial_w[:]
    for i in range(max_iters):
        grad = np.matmul(tx.T, sigmoid(tx.dot(w) - y)) + (lambda_*w)
        w = w - gamma * grad
    loss = grad
    return loss, w

'''
def logistic_regression(y,tx,initial_w,max_iters,gamma):
    w = initial_w[:]
    for i in range(max_iters):
        e = y - sigmoid(tx.dot(w))
        grad = - tx.T.dot(e) / len(e)
        w = w - gamma * grad
    loss = calculate_mse(e)
    return loss, w
    

def logistic_regression_sgd(y,tx,initial_w,batch_size,max_iters,gamma):
    w = initial_w[:]
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            e = y - sigmoid(tx.dot(w))
            grad = - tx.T.dot(e) / len(e)
            w = w - gamma * grad
    loss = calculate_mse(e)
    return loss, w
'''



"""
TRAINING 
"""

y,x,i = load_csv_data('data/train.csv',sub_sample=False)

# next step standardize X
x, mean_x, std_x = standardize(x) 
# Add one dimenssion to X with only 1,beacause 1*W0+ x1*W1 + ...
b = np.ones((x.shape[0],1), dtype = int)
x = np.column_stack((b, x)) 
# Now creat vector Polynomial basis
#x = build_poly(x,31 )
initial_w = np.random.rand(x.shape[1],1)
y = y.reshape(y.shape[0],1)

#loss, w = least_squares_GD(y,x,initial_w,1000,0.001)
#loss , w = least_squares_SGD(y,x,initial_w,1,500,0.001)
#loss, w = least_squares(y, x)
#loss, w = logistic_regression(y,x,initial_w,1000,0.01)
loss, w = reg_logistic_regression(y,x,initial_w,1000,0.01,.1)
print(loss,w)


y,x,i = load_csv_data('data/train.csv',sub_sample=False)
x, mean_x, std_x = standardize(x) 
# Add one dimenssion to X with only 1,beacause 1*W0+ x1*W1 + ...
b = np.ones((x.shape[0],1), dtype = int)
x = np.column_stack((b, x)) 

y_test = predict_labels(w, x)
n = 0
for i in range(len(y)-1):
    if (y[i] == y_test[i]): 
        n += 1
        
print("Prediction accurracy: " )
print(n/len(y))
