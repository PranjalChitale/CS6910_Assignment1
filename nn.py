import numpy as np
from activation_fn import *
from weight_init import random_init

class Layer(object):
    def __init__(self, num_neurons, prev_neurons, activation_func, weight_init_method, batch_size):
        self.weights = eval(weight_init_method + "(num_neurons, prev_neurons)")     
        self.bias  = np.zeros((1, num_neurons))
        self.activation_func = activation_func
        self.a = None #np.zeros((batch_size, num_neurons)) #pre-activation
        self.h = None #np.zeros((batch_size, num_neurons)) #activation
        self.grad_w = None
        self.grad_b = None
        self.grad_a = None
        self.grad_h = None

class FFN(object):
    def __init__(self, input_dim, no_classes, hidden_layer_neurons, activation_func, weight_init_layers, batch_size):
        """
        input_dim: Dimensions of the input layer.
        no_classes : No of neurons in output layer.
        hidden_layer_neurons : A list containing no of neurons
                               at each layer.
                               len(hidden_layer_neurons) = no of hidden layers.

        weight_init_layers : A list containing weight initialization strategy at each layer.

        """
        neurons = [input_dim] + hidden_layer_neurons + [no_classes]
        self.layers = [Layer(neurons[i+1], neurons[i], activation_func[i], weight_init_layers[i], batch_size) for i in range(len(neurons) - 1)]

    def forward_propogation(self, x):
        for i in range(0, len(self.layers)):
            if i == 0:
                self.layers[i].a = np.dot(x, self.layers[i].weights) + self.layers[i].bias  
            else:
                self.layers[i].a = np.dot(self.layers[i-1].h, self.layers[i].weights) + self.layers[i].bias

            self.layers[i].h =  eval(self.layers[i].activation_func + "(self.layers[i].a)")         
