import numpy as np
from nn import *
from metrics import *

def vgd(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.001, batch_size = 1):
    '''
    Implements the vanilla version of the Gradient Descent Algorithm
    If bacth size is set to 1, acts like the Stochiastic version. 
    If batch size is set, acts like the Mini-batch version.
    If batch size is set to input.shape[0], batch gradient descent.
    '''
    best_model, best_acc = 0,0
    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]
            batch_pred = ffn.forward_propogation(batch_images)
            ffn.backward_propogation(batch_labels, batch_pred)  

            for k in range(0, len(ffn.layers)):
                ffn.layers[k].weights = ffn.layers[k].weights - (lr * ffn.layers[k].grad_w + lr * ffn.weight_decay*ffn.layers[k].weights)
                ffn.layers[k].bias = ffn.layers[k].bias -  (lr *ffn.layers[k].grad_b)


        y_hat = ffn.get_prediction(train_images)
        

        train_loss = eval(ffn.loss_fn + '(train_labels, y_hat)')     

        train_pred = ffn.get_prediction(train_images)
        train_acc = accuracy(train_labels, train_pred) 

        if val != None:
            val_images, val_labels = val 
            val_pred = ffn.get_prediction(val_images)
            val_acc = accuracy(val_labels, val_pred)
            val_loss = eval(ffn.loss_fn + '(val_labels, val_pred)')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = ffn

        print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
        print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))

    print(best_acc)
    return best_model





def mgd(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, gamma = 0.9):
    '''
    Implements Momentum based Gradient Descent.
    '''
    best_model, best_acc = 0,0
    prev_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    prev_B = [np.zeros_like(layer.bias) for layer in ffn.layers]

    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]
            batch_pred = ffn.get_prediction(batch_images)
            ffn.backward_propogation(batch_labels, batch_pred) 

            for k in range(0, len(ffn.layers)):
                if j==0:
                    ffn.layers[k].weights = ffn.layers[k].weights -  (lr * ffn.layers[k].grad_w + lr * ffn.weight_decay*ffn.layers[k].weights)
                    ffn.layers[k].bias = ffn.layers[k].bias -  (lr *ffn.layers[k].grad_b)
                    prev_W[k] = lr*ffn.layers[k].grad_w +  lr * ffn.weight_decay*ffn.layers[k].weights
                    prev_B[k] = lr *ffn.layers[k].grad_b
                else:
                    prev_W[k] = np.multiply(gamma, prev_W[k]) + lr*ffn.layers[k].grad_w +  lr * ffn.weight_decay*ffn.layers[k].weights
                    prev_B[k] = np.multiply(gamma, prev_B[k]) + lr *ffn.layers[k].grad_b
                    ffn.layers[k].weights = ffn.layers[k].weights - prev_W[k]
                    ffn.layers[k].bias = ffn.layers[k].bias - prev_B[k]


        y_hat = ffn.get_prediction(train_images)
        
        train_loss = eval(ffn.loss_fn + '(train_labels, y_hat)')     

        train_pred = ffn.get_prediction(train_images)
        train_acc = accuracy(train_labels, train_pred) 

        if val != None:
            val_images, val_labels = val 
            val_pred = ffn.get_prediction(val_images)
            val_acc = accuracy(val_labels, val_pred)
            val_loss = eval(ffn.loss_fn + '(val_labels, val_pred)')  
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = ffn

        print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
        print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))

    print(best_acc)
    return best_model

