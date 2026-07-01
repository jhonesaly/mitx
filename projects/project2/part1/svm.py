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
    svm_model = LinearSVC(random_state=0, C=0.1)
    svm_model.fit(train_x, train_y)
    pred_test_y = svm_model.predict(test_x)
    
    return pred_test_y


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

