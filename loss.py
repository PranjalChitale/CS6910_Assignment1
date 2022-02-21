import numpy as np
def cross_entropy(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    temp = np.multiply(y_true, y_hat) #Correct class's probability multiplied by 1 rest 0
    loss =  np.average(-np.log(np.sum(temp, axis = 1) + 1e-8)) #Smoothing to prevent log from blowing up.
    return loss

def squared_error(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    loss = np.mean(np.square(y_hat - y_true))
    return loss

def cross_entropy_grad(y_true, y_hat):
    grad_ce = y_hat - y_true
    return grad_ce

##TODO ##
def squared_error_grad(y_true, y_hat):
    temp = y_hat - y_true
    diff = np.multiply(y_true, y_hat).sum(axis = 1, keepdims = True)
    return (temp - diff)*y_hat