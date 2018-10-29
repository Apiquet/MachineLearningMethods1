Hello!
Welcome to the README for team PES for the EPFL Fall 2018 Machine Learning Class Project 1.

author: SMARDA/ERBACHER/PIQUET

PYTHON 3.

Used libraries:
-Numpy
-CSV

********************************************************************************
To generate the csv submission file with predictions for the input test data,
simply execute run.py, such as with:

$ python3 run.py

Notes:
-run.py doesn't need argument.
-run.py is using by default the ridge_regression algorithm
********************************************************************************

Additional Notes:

The implementation.py and proj1_helpers.py contains all the useful functions.

To test the other training algorithms besides the default ridge_regression,
change the argument 'FUNCTION' at the bottom of the run.py file:
FUNCTION = 0 => Least_squares_GD
FUNCTION = 1 => Least_squares_SGD
FUNCTION = 2 => Least_squares
FUNCTION = 3 => ridge_regression
FUNCTION = 4 => logistic_regression
FUNCTION = 5 => reg_logistic_regression

GAMMA   : Learning rate for gradient descent algorithms
MAX_ITER: Number of iterations for gradient descent algorithms
LAMBDA_ : Regularization parameter

Pre-processing:
We append ones to the input dataset and remove columns with too many outliers.
We also replace outliers with column mean values in the cross-validation folds.
The build poly is set to order 1. Polynomial over 1 it makes the least_square function overflow :(

Crossvalidation:
The accuracy is computed with the crossvalidation function (further details can
be found in the implementations.py file).
By default we train on 9/10 of the train-set
we test our accuracy with the 1/10 of the train-set.

Label generation:
Because the sigmoid function return 0 or 1, we transformed the -1 values of Y vector to 0
when reading in the training data.
We reverse this before creating the submission sample, using the function
'predict_reverse' (also in the implementations.py file)

Have a great day!
