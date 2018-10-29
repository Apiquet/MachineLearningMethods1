# -*- coding: utf-8 -*-
"""
@author: SMARDA/ERBACHER/PIQUET
"""

import implementations
import numpy as np

"""
Changing the argument 'function' specifies which training algorithm to use:

function = 0 => Least_squares_GD
function = 1 => Least_squares_SGD
function = 2 => Least_squares
function = 3 => ridge_regression
function = 4 => logistic_regression
function = 5 => reg_logistic_regression
"""

MAX_ITER = 1000
GAMMA = 0.01
LAMBDA_ = 2
FUNCTION = 5

params = [MAX_ITER, GAMMA, LAMBDA_, FUNCTION]

implementations.main(params)
