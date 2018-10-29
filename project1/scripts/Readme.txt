
author: SMARDA/ERBACHER/PIQUET
Machine learning project

The implementation.py and proj1_helpers.py contains all the usefull function.

The run.py doesn't need argument.
run.py is using by default the leastSquareGD


To test the other function:

change the argument 'function':
function = 0 => Least_squares_GD
function = 1 => Least_squares_SGD
function = 2 => Least_squares
function = 3 => ridge_regression
function = 4 => logistic_regression
function = 5 => reg_logistic_regression

The accuracy is the accuracy with the crossvalidation.

By default the we train on 9/10 of the train-set 
we test our accuracy with the 1/10 of the train-set

The run.py generate the sample-submission.csv based on the test.cvs data-set.