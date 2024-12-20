import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def compute_logistic_cost_regularized(X, y, theta, lambda_):
    m = len(y)
    X = np.c_[np.ones((m, 1)), X]  # Add intercept term
    h = sigmoid(X @ theta)
    cost = -(1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    
    return cost + reg_term

def gradient_descent_logistic(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    X = np.c_[np.ones((m, 1)), X]  # Add intercept term
    J_history = []

    for _ in range(num_iters):
        h = sigmoid(X @ theta)  # Compute the hypothesis
        error = h - y  # Error vector
        grad = (1 / m) * (X.T @ error)  # Gradient for all parameters
        reg_term = (lambda_ / m) * np.r_[0, theta[1:]]  # Regularization term (excluding intercept)
        theta -= alpha * (grad + reg_term)  # Update parameters
        J_history.append(compute_logistic_cost_regularized(X[:, 1:], y, theta, lambda_))

    return theta, J_history












