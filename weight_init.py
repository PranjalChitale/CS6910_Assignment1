import numpy as np
import math
np.random.seed(0)

def random_init(prev_neurons, num_neurons):
    '''
    Random initialization of weights and bias.
    '''
    weights = np.random.randn(prev_neurons, num_neurons) 
    bias = np.random.randn(1, num_neurons)
    return weights, bias

def xavier_init(prev_neurons, num_neurons):
    '''
    Based on the paper by Dr. Xavier Glorot & Dr. Yoshua Bengio
    '''
    lower_limit, upper_limit = -math.sqrt(6.0/(num_neurons + prev_neurons)), math.sqrt(6.0/(num_neurons + prev_neurons)) 
    weights = np.random.uniform(lower_limit, upper_limit, size=(prev_neurons, num_neurons))
    bias = np.random.uniform(lower_limit, upper_limit, size=(1, num_neurons))
    return weights, bias

def kaiming_init(prev_neurons, num_neurons):
    '''
    Based on the paper by Dr. Kaiming He.
    '''
    sd = math.sqrt(2.0 / prev_neurons)
    weights = np.random.randn(prev_neurons, num_neurons)*math.sqrt(2/prev_neurons)
    bias = np.random.randn(1, num_neurons)
    return weights, bias
