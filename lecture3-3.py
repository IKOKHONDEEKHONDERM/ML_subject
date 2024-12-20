import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_cost_regularized(X, y, theta, lambda_):
    m = len(y)  
    X = np.c_[np.ones((m, 1)), X] 
    h = sigmoid(X @ theta) 
    cost = -(1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    J = cost + reg_term
    
    return J

