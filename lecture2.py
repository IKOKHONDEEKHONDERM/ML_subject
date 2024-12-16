import numpy as np

def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

def logistic_regression_predict(X, theta):
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    z = np.dot(X_with_intercept, theta)
    
    return sigmoid(z)