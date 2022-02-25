#Imports
from keras.datasets import fashion_mnist
import numpy as np
from preprocess_f_mnist import preprocess
from nn import FFN
from loss import *
from metrics import accuracy

#Getting the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Define the val size.
val_size = 0.10 

#Split the data
train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

#Create a Model
model = FFN(input_dim = 784, num_classes = 10, optimizer="vgd",weight_decay=0.001, loss_fn="cross_entropy")
model.add_layer(256,weight_init_method="kaiming_init",activation_func="relu")
model.add_layer(128,weight_init_method="kaiming_init",activation_func="relu")
model.add_layer(64,weight_init_method="kaiming_init",activation_func="relu")
model.add_layer(32,weight_init_method="kaiming_init",activation_func="relu")
model.add_layer(10,weight_init_method="xavier_init",activation_func="linear")

#Train the Model
model.train(train_images, train_labels, val = (val_images, val_labels), learning_rate= 0.0001, batch_size=128,epochs=10)
