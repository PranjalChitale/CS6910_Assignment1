#Imports
import numpy as np

#Activation functions

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def relu(x):
	return np.max(0,x)

def softmax(x):
	"""
	The formula for softmax is e_i^x/sum_j(e_j^x)
	However, this function is susceptible to underflow and overflow.
	To avoid this, we follow the method suggested in the Deep Learning book by Dr. Goodfellow et al.
	https://www.deeplearningbook.org/contents/numerical.html
	We set z = x - max(x) and then compute the softmax of z.
	"""
	z = x - np.max(x, axis = 1, keepdims = True) #Subtract max of each row. 
	exp = np.exp(z) #Exponentiation
	sum_exp = np.sum(exp, axis = 1, keepdims = True) #Sum of Exponentiated version.
	return exp / sum_exp 

#Derivative of Activation functions

def sigmoid_derivative(x):
	return np.multiply(x, 1-x)

def tanh_derivative(x):
	return 1 - np.square(x)
    
def relu_derivative(x):
	x_prime = np.zeros_like(x)
	for i in range(0,len(x)):
		if x[i]>0:
			x_prime[i] = 1
	return x_prime
