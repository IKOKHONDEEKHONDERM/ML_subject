# Function to compute the logistic regression cost
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_logistic_cost(X, y, theta):
   
    m = len(y)
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  # Corrected np.one -> np.ones

    z = np.dot(X_with_intercept, theta)
    y_hat = sigmoid(z)

    cost = -(1 / m) * (np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat)))

    return cost
