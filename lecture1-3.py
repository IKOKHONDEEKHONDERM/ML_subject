import numpy as np

def gradient_descent(X, y, theta, alpha, num_iters):
   
    theta = theta.astype(float)  
    m = len(y) 
    J_history = []  
    
    X_intercept = np.hstack((np.ones((m,1)),X))

    for i in range(num_iters):
        predictions = np.dot(X_intercept, theta)
        error = predictions - y
        gradient = (1/m) * np.dot(X_intercept.T, error)
        theta -= alpha * gradient
        cost = (1 / (2 * m)) * np.sum(error ** 2)

        J_history.append(cost)
        
    return theta, J_history