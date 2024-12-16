import numpy as np

def compute_r2(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_mean)**2)
    
    if ss_tot == 0:
        return 1.0 
        
    #คำนวนหาค่า r2
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

