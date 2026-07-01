import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation
    """
    # Inicializa o modelo com os parâmetros exigidos
    svm_model = LinearSVC(random_state=0, C=0.1)
    
    # Treina o modelo com os dados fornecidos
    svm_model.fit(train_x, train_y)
    
    # Realiza a predição nos dados de teste e retorna o resultado
    pred_test_y = svm_model.predict(test_x)
    
    return pred_test_y


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

