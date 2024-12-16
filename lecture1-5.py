import numpy as np

def normal_equation(X, y):

    # เพิ่ม Intercept term (คอลัมน์ของค่า 1) เข้าไปใน X
    m = X.shape[0]
    X_aug = np.hstack([np.ones((m, 1)), X])  # X_aug เป็นเมทริกซ์ m x (n+1)
    
    # ใช้ pseudo-inverse แทน inverse เพื่อแก้ปัญหา singular matrix
    theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
    
    return theta