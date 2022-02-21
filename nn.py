import numpy as np
from activation_fn import *
from weight_init import *
from loss import *
from metrics import *
from optimizers import *

np.random.seed(0)

class Layer(object):
    def __init__(self, num_neurons, prev_neurons, activation_func, weight_init_method):
        self.weights, self.bias = eval(weight_init_method + "(prev_neurons, num_neurons)")     
        self.activation_func = activation_func
        self.input = None
        self.a = None # At run-time, shape is (batch_size, num_neurons) #pre-activation
        self.h = None # At run-time, shape is (batch_size, num_neurons) #activation
        self.grad_w = None #Shape is same as that of the weights.
        self.grad_b = None #Shape is same as that of the bias.
        self.grad_a = None #At run-time, shape is (batch_size, num_neurons)
        self.grad_h = None #At run-time, shape is (batch_size, num_neurons)

    def forward_pass(self, x):
        self.input = x
        self.a = self.input @ self.weights + self.bias
        self.h = eval(self.activation_func + "(self.a)")
        return self.h
    
    def backward_pass(self, next_grad, weight_decay):
        self.grad_h = next_grad
        self.grad_a = np.multiply(eval(self.activation_func + "_derivative" + "(self.a)"), next_grad) 
        self.grad_w = self.input.T @ self.grad_a + weight_decay * self.weights
        self.grad_b = np.sum(self.grad_a, axis = 0) + weight_decay*self.bias
        return self.grad_a @ self.weights.T

class FFN(object):
    def __init__(self, input_dim, num_classes, loss_fn, optimizer, weight_decay = 0):
        """
        input_dim: Dimensions of the input layer.
        no_classes : No of neurons in output layer.
        Optimizer: Optimization Algorithm to be used.
        loss_fn : Loss function to be used.
        weight_decay : Species the value (Lambda) used for weight decay, default value is 0.
        """
        self.optimizer = optimizer 
        self.input_dim = input_dim
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.layers = []

    def add_layer(self, num_neurons, activation_func, weight_init_method):
        if len(self.layers) == 0:
            prev_neurons = self.input_dim #This handles input layer
        else:
            prev_neurons = self.layers[-1].weights.shape[1] #This handles other layers.
        self.layers.append(Layer(num_neurons, prev_neurons, activation_func, weight_init_method))
        
    def forward_propogation(self, x):
        '''
        Takes the input x and propogates it through all the layers.  
        '''
        temp = x
        for i in range(0, len(self.layers)):
            temp = self.layers[i].forward_pass(temp)
        return softmax(temp)      

    def backward_propogation(self, y_true, y_pred):
        '''
        First calculates the gradient of the loss function.
        And back propogates it by layer by layer till the input
        '''
        loss_grad =  eval(self.loss_fn + "_grad" + "(y_true, y_pred)") 
        next_grad = loss_grad
        for i in range(len(self.layers) - 1, -1, -1):
            next_grad = self.layers[i].backward_pass(next_grad, self.weight_decay)
            
    def get_prediction(self, x):
        y_hat = self.forward_propogation(x)
        return y_hat

    def reset_grad(self):
        for i in range(0,len(self.layers)):
            self.layers[i].grad_w = None
            self.layers[i].grad_b = None
            self.layers[i].grad_a = None
            self.layers[i].grad_h = None

    def train(self, train_images, train_labels, val, learning_rate = 0.001, batch_size = 128, epochs = 1, beta_1 = 0.9, beta_2 = 0.999, gamma = 0.9):
        if self.optimizer == "vgd":
            vgd(self, train_images, train_labels, val = val, epochs = epochs , lr = learning_rate, batch_size = batch_size)
        elif self.optimizer == "mgd":
            mgd(self, train_images, train_labels, val = val, epochs = epochs , lr = learning_rate, batch_size = batch_size, gamma = 0.9)
          