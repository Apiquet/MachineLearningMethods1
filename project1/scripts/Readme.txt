
author: SMARDA/ERBACHER/PIQUET
Machine learning project
PYTHON 3.

Used library: -Numpy
	      -CSV


-run.py doesn't need argument.
-run.py is using by default the ridge_regression


The implementation.py and proj1_helpers.py contains all the usefull function.



To test the other function:

change the argument 'function':
function = 0 => Least_squares_GD
function = 1 => Least_squares_SGD
function = 2 => Least_squares
function = 3 => ridge_regression
function = 4 => logistic_regression
function = 5 => reg_logistic_regression

Gamma   : Learning rate for gradient descent algorithms
Max_iter: Number of iterations for gradient descent algorithms
Lambda  : Regularization parameter


Crossvalidation : 		
The accuracy is computed with the crossvalidation.
By default the we train on 9/10 of the train-set 
we test our accuracy with the 1/10 of the train-set

The run.py generate the sample-submission.csv based on the test.cvs data-set.

Because the sigmoid function return 0 or 1, we transformed the -1 values of Y vector to 0.
We make the reverse function before creating the submission sample, with the function 'predict_reverse'.

The build poly is set to order 1. because over 1 it makes the least_square function overflow :(

The ridge regression also give the same result for different value of lambda.
