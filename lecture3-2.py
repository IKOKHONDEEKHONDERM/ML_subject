import numpy as np

def compute_cost_regularized(X, y, theta, lambda_):
  
    m = len(y)
    X = np.c_[np.ones((m, 1)), X]
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term

def gradient_descent_linear(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    X = np.c_[np.ones((m, 1)), X]
    J_history = []
    
    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        grad = (1 / m) * (X.T @ errors)
        reg_term = (lambda_ / m) * np.r_[[0], theta[1:]]
        grad += reg_term
        theta -= alpha * grad
        cost = compute_cost_regularized(X[:, 1:], y, theta, lambda_)
        J_history.append(cost)
    
    return theta, np.array(J_history)













