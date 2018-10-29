# -*- coding: utf-8 -*-
"""
@author: SMARDA/ERBACHER/PIQUET
"""

import implementations
import numpy as np

"""
change the argument 'function':
function = 0 => Least_squares_GD
function = 1 => Least_squares_SGD
function = 2 => Least_squares
function = 3 => ridge_regression
function = 4 => logistic_regression
function = 5 => reg_logistic_regression


By default the we train on 9/10 of the train-set 
we test our accuracy with the 1/10 of the train-set

"""

Max_iter = 1000


Gamma = 0.01
lambda_ = 2
function = 5

param = [Max_iter,Gamma,lambda_,function]

implementations.main(param)
