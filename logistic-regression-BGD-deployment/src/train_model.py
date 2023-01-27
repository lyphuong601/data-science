import numpy as np
import copy
import math

##########################################################
# Regression routine
##########################################################
def sigmoid(z):
    z = np.clip( z, -500, 500 )                     # protect against overflow
    g = 1.0/(1.0+np.exp(-z))
    return g


def compute_cost_matrix(X, y, w, b, lambda_=0, safe=True):
    m = X.shape[0]
    f    = sigmoid(X @ w + b)                       # (m,n)(n,1) = (m,1)
    cost = (1/m)*np.sum((np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))) + (lambda_/ (2*m))  # (1,m)(m,1) = (1,1)
    return cost   


def compute_gradient_matrix(X, y, w, b, lambda_=0):
    m, n = X.shape

    f_wb  = sigmoid( X @ w + b )                    # (m,n)(n,1) = (m,1)
    err   = np.subtract(f_wb, y.values)             # (m,1)
    dj_dw = (1/m) * (X.T @ err) + (lambda_/m) * w   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                     # scalar      
    
    return dj_db, dj_dw 


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    J_history = []
    w = copy.deepcopy(w_in) 
    b = b_in

    for i in range(num_iters):

        dj_db, dj_dw = compute_gradient_matrix(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:                                # prevent resource exhaustion
            J_history.append(compute_cost_matrix(X, y, w, b, lambda_) )
        
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    
    return w, b, J_history                          #return final w,b and J history for graphing

