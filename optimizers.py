import numpy as np
from nn import *
from metrics import *
import copy
import wandb
import pickle

def vgd(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.001, batch_size = 1):
    '''
    Reference :- http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf
    Implements the vanilla version of the Gradient Descent Algorithm
    If batch size is set to 1, acts like the Stochiastic version.
    If batch size smaller than input.shape[0] is set, acts like the Mini-batch version. 
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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model


def mgd(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, gamma = 0.9):
    '''
    Implements Momentum based Gradient Descent.
    Reference :- http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf
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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model


def nag(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, gamma = 0.9):
    '''
    Implements Nesterov Accelrated Gradient Descent.
    Reference :- http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf
    '''
    best_model, best_acc = 0,0
    prev_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    prev_B = [np.zeros_like(layer.bias) for layer in ffn.layers]

    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]

            #Look ahead
            ffn_temp = copy.deepcopy(ffn)
            #Take a step based on history.
            if j!=0:
                for k in range(0, len(ffn_temp.layers)):
                    ffn_temp.layers[k].weights = ffn_temp.layers[k].weights - np.multiply(gamma, prev_W[k])
                    ffn_temp.layers[k].bias = ffn_temp.layers[k].bias - np.multiply(gamma, prev_B[k])
            
            #Calculate look ahead gradients.
                batch_pred = ffn_temp.get_prediction(batch_images)
                ffn_temp.backward_propogation(batch_labels, batch_pred) 

            for k in range(0, len(ffn.layers)):
                if j==0:
                    batch_pred = ffn.get_prediction(batch_images)
                    ffn.backward_propogation(batch_labels, batch_pred) 
                    ffn.layers[k].weights = ffn.layers[k].weights -  (lr * ffn.layers[k].grad_w + lr * ffn.weight_decay*ffn.layers[k].weights)
                    ffn.layers[k].bias = ffn.layers[k].bias -  (lr *ffn.layers[k].grad_b)
                    prev_W[k] = lr*ffn.layers[k].grad_w +  lr * ffn.weight_decay*ffn.layers[k].weights
                    prev_B[k] = lr *ffn.layers[k].grad_b
                else:
                    prev_W[k] = np.multiply(gamma, prev_W[k]) + lr*ffn_temp.layers[k].grad_w +  lr * ffn.weight_decay*ffn.layers[k].weights
                    prev_B[k] = np.multiply(gamma, prev_B[k]) + lr *ffn_temp.layers[k].grad_b
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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model



def rmsprop(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, beta_1 = 0.9):
    '''
    Implements RMSProp Gradient Descent.
    Reference :- http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf
    '''
    best_model, best_acc = 0,0
    eps = 1e-8
    v_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    v_B = [np.zeros_like(layer.bias) for layer in ffn.layers]

    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]

            batch_pred = ffn.get_prediction(batch_images)
            ffn.backward_propogation(batch_labels, batch_pred) 

            for k in range(0, len(ffn.layers)):
                
                v_W[k] = np.multiply(beta_1, v_W[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_w**2) 
                v_B[k] = np.multiply(beta_1, v_B[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_b**2)
                ffn.layers[k].weights = ffn.layers[k].weights - (lr / np.sqrt(v_W[k] + eps)) * ffn.layers[k].grad_w
                ffn.layers[k].bias = ffn.layers[k].bias - (lr / np.sqrt(v_B[k] + eps)) * ffn.layers[k].grad_b


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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model


def adam(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, beta_1 = 0.9, beta_2 = 0.999):
    '''
    Implements Adam Gradient Descent.
    Reference :- http://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Handout/Lecture5.pdf
    '''
    best_model, best_acc = 0,0
    eps = 1e-8
    v_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    v_B = [np.zeros_like(layer.bias) for layer in ffn.layers]
    v_W_hat = [np.zeros_like(layer.weights) for layer in ffn.layers]
    v_B_hat = [np.zeros_like(layer.bias) for layer in ffn.layers]
    m_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    m_B = [np.zeros_like(layer.bias) for layer in ffn.layers]
    m_W_hat = [np.zeros_like(layer.weights) for layer in ffn.layers]
    m_B_hat = [np.zeros_like(layer.bias) for layer in ffn.layers]

    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]

            batch_pred = ffn.get_prediction(batch_images)
            ffn.backward_propogation(batch_labels, batch_pred) 

            for k in range(0, len(ffn.layers)):
                
                m_W[k] = np.multiply(beta_1, m_W[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_w) 
                m_B[k] = np.multiply(beta_1, m_B[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_b)

                v_W[k] = np.multiply(beta_2, v_W[k]) + np.multiply(1 - beta_2, ffn.layers[k].grad_w**2) 
                v_B[k] = np.multiply(beta_2, v_B[k]) + np.multiply(1 - beta_2, ffn.layers[k].grad_b**2)
                
                m_W_hat[k] =  m_W[k] / (1 - math.pow(beta_1, i+1))
                m_B_hat[k] =  m_B[k] / (1 - math.pow(beta_1, i+1))

                v_W_hat[k] =  v_W[k] / (1 - math.pow(beta_2, i+1))
                v_B_hat[k] =  v_B[k] / (1 - math.pow(beta_2, i+1))
                
                ffn.layers[k].weights = ffn.layers[k].weights - (lr / np.sqrt(v_W_hat[k] + eps)) * m_W_hat[k]
                ffn.layers[k].bias = ffn.layers[k].bias - (lr / np.sqrt(v_B_hat[k] + eps)) * m_B_hat[k]

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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model


def nadam(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.01, batch_size = 1, beta_1 = 0.9, beta_2 = 0.999):
    '''
    Implements Nestrov Accelrated Adam Gradient Descent.
    Update rule Reference : https://ruder.io/optimizing-gradient-descent/index.html#nadam
    '''
    best_model, best_acc = 0,0
    eps = 1e-8
    v_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    v_B = [np.zeros_like(layer.bias) for layer in ffn.layers]
    v_W_hat = [np.zeros_like(layer.weights) for layer in ffn.layers]
    v_B_hat = [np.zeros_like(layer.bias) for layer in ffn.layers]
    m_W = [np.zeros_like(layer.weights) for layer in ffn.layers]
    m_B = [np.zeros_like(layer.bias) for layer in ffn.layers]
    m_W_hat = [np.zeros_like(layer.weights) for layer in ffn.layers]
    m_B_hat = [np.zeros_like(layer.bias) for layer in ffn.layers]

    for j in range(epochs):
        for i in range(0, train_images.shape[0], batch_size):
            #Pick a batch of examples to train.
            batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
            batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]

            batch_pred = ffn.get_prediction(batch_images)
            ffn.backward_propogation(batch_labels, batch_pred) 

            for k in range(0, len(ffn.layers)):
                
                m_W[k] = np.multiply(beta_1, m_W[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_w) 
                m_B[k] = np.multiply(beta_1, m_B[k]) + np.multiply(1 - beta_1, ffn.layers[k].grad_b)

                v_W[k] = np.multiply(beta_2, v_W[k]) + np.multiply(1 - beta_2, ffn.layers[k].grad_w**2) 
                v_B[k] = np.multiply(beta_2, v_B[k]) + np.multiply(1 - beta_2, ffn.layers[k].grad_b**2)
                
                m_W_hat[k] =  m_W[k] / (1 - math.pow(beta_1, i+1))
                m_B_hat[k] =  m_B[k] / (1 - math.pow(beta_1, i+1))

                v_W_hat[k] =  v_W[k] / (1 - math.pow(beta_2, i+1))
                v_B_hat[k] =  v_B[k] / (1 - math.pow(beta_2, i+1))
                
                b = (1-beta_1)/(1-math.pow(beta_1,i+1))

                ffn.layers[k].weights = ffn.layers[k].weights - (lr / np.sqrt(v_W_hat[k] + eps)) * (beta_1 * m_W_hat[k] + b*ffn.layers[k].grad_w)
                ffn.layers[k].bias = ffn.layers[k].bias - (lr / np.sqrt(v_B_hat[k] + eps)) * (beta_1 * m_B_hat[k] + b*ffn.layers[k].grad_b)

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
                best_model = copy.deepcopy(ffn)
                with open('model_pkl', 'wb') as files:
                    pickle.dump(ffn, files)

            print("Epoch {} completed, training_loss = {}, validation_loss = {}.".format(j, train_loss, val_loss))
            print("Training accuracy = {}, Validation Accuracy = {}".format(train_acc, val_acc))
            wandb.log({"train_acc": train_acc,"val_acc": val_acc,"train_loss": train_loss,"val_loss": val_loss})

    print(best_acc)
    return best_model

