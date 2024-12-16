import numpy as np

def linear_regression_predict(X, theta):
    m = X.shape[0]
    X_intercept = np.hstack((np.ones((m, 1)), X))  
    y_predict = np.dot(X_intercept, theta) 
    
    return y_predict
 