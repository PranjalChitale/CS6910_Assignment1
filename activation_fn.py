#Imports
import numpy as np

#Activation functions

def sigmoid(x):
    #In case of Sigmoid, Math overflow was observed so clipping the inputs. 
    x = np.clip(x, a_min = -10**3, a_max = 10*3)
    return (1/(1+ np.exp(-x)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    #Sets negative values to zero, rest unchanged
    x[x<0] = 0
    return x

def linear(x):
    '''
    Added this for convenience to maintain the backprop loop.
    Since, we use the shortcut formula (y_hat - y_true) for loss gradient directly wrt output layer.
    '''
    return x

def softmax(x):
    '''
    The formula for softmax is e_i^x/sum_j(e_j^x)
    However, this function is susceptible to underflow and overflow.
    To avoid this, we follow the method suggested in the Deep Learning book by Dr. Goodfellow et al.
    https://www.deeplearningbook.org/contents/numerical.html
    We set z = x - max(x) and then compute the softmax of z.
	'''
    z = x - np.max(x, axis = 1, keepdims = True) #Subtract max of each row. 
    exp = np.exp(z) #Exponentiation
    sum_exp = np.sum(exp, axis = 1, keepdims = True) #Sum of Exponentiated version.
    return exp / sum_exp 

#Calculate derivative of Activation functions to be used in the Backpropogation stage.
def relu_derivative(x):
    '''
    For negative values, the derivative is zero, for positive it is 1.
    '''
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid_derivative(a):
    '''
    Derivative of f(x) = sigmoid(X) results into f(x)(1-f(x))
    '''
    h = sigmoid(a)
    return h*(1-h)
  
def tanh_derivative(a):
    '''
    Derivative of f(x) = tanh(X) results into 1-f(x)^2
    '''
    h = tanh(a)
    return 1 - h*h

def linear_derivative(a):
    """
    Derivative is one.
    """
    return np.ones(a.shape)

