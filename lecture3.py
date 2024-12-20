import numpy as np

def compute_cost_regularized(X, y, theta, lambda_):
    m = len(y)
    X = np.c_[np.ones((m, 1)), X]

    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    total_cost = cost + reg_term
    
    return total_cost

