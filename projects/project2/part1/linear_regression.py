import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization
    """
    # Cria a matriz identidade com a mesma dimensão do número de features (d + 1)
    I = np.eye(X.shape[1])
    
    # Calcula theta usando a fórmula de forma fechada
    theta = np.linalg.inv(X.T @ X + lambda_factor * I) @ X.T @ Y
    
    return theta

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
