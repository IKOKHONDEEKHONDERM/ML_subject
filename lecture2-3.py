import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def compute_logistic_cost(X, y, theta):
    
    m = len(y)
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    z = np.dot(X_with_intercept, theta)
    y_hat = sigmoid(z)
    cost = -(1 / m) * (np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat)))
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []  
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    
    for i in range(num_iters):
        z = np.dot(X_with_intercept, theta)
        y_hat = sigmoid(z)
        gradient = (1 / m) * np.dot(X_with_intercept.T, (y_hat - y))
        theta -= alpha * gradient
        cost = compute_logistic_cost(X, y, theta)
        J_history.append(cost)

    return theta, J_history










