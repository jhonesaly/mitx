import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    I = np.eye(X.shape[1])
    theta = np.linalg.solve(X.T @ X + lambda_factor * I, X.T @ Y)
    return theta

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
