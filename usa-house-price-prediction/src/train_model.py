import numpy as np
import copy
import math

##########################################################
# Regression Routines
##########################################################

def compute_cost(X, y, w, b, lambda_=10):
    m = X.shape[0]
    f_wb = X @ w + b
    err  = np.subtract(f_wb, y.values)  # (m,1)
    cost = (1/(2*m)) * np.sum(err**2) + (lambda_/ (2*m)) * np.sum(w**2)
    return cost


def compute_gradient(X, y, w, b, lambda_=10):
    m = X.shape[0]
    f_wb = X @ w + b  # (m,n)(n,1) = (m,1)
    err   = np.subtract(f_wb, y.values)  # (m,1)
    dj_dw  = (1/m) * (X.T @ err) + (lambda_/m) * w  # (n,m)(m,1) = (n,1)
    dj_db  = (1/m) * np.sum(err)   # scalar   
    
    return dj_db, dj_dw

  
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        dj_db, dj_dw = compute_gradient(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:      # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b, lambda_) )
        
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    
    return w, b, J_history  #return final w,b and J history for graphing


##########################################################
# Normalization Routines
##########################################################

def zscore_normalize_features(X, rtn_ms=False):
    mu     = np.mean(X, axis=0)  
    sigma  = np.std(X, axis=0) 
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
        