import numpy as np
#Loss Functions
def cross_entropy(y_true, y_hat):
    """
    Assumes, y_true is one-hot encoded.
    """
    #Correct class's probability multiplied by 1 rest 0
    temp = np.multiply(y_true, y_hat) 
    #Smoothing to prevent log from blowing up.
    loss =  np.average(-np.log(np.sum(temp, axis = 1) + 1e-8)) 
    return loss

def squared_error(y_true, y_hat):
    """
    Assumes, y_true is one-hot encoded.
    """
    loss = np.mean(np.square(y_hat - y_true))
    return loss

#Gradients of the Loss Functions
def cross_entropy_grad(y_true, y_hat):
    grad_ce = y_hat - y_true
    return grad_ce

def squared_error_grad(y_true, y_hat):
    diff = y_hat - y_true
    temp = np.multiply(y_true, y_hat).sum(axis = 1, keepdims = True)
    return (diff - temp)*y_hat