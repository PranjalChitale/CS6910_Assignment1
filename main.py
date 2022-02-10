#Imports
from keras.datasets import fashion_mnist
import numpy as np
from preprocess_f_mnist import preprocess
from nn import FFN
from loss import *
from metrics import accuracy

#Getting the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

val_size = 0.10 

#train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

train_images, train_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

ffn = FFN(784,10,[128,64,32], ["tanh", "sigmoid", "tanh","softmax"], ["random_init", "random_init", "random_init","random_init"], train_images.shape[0])

ffn.forward_propogation(train_images)
y_hat = ffn.layers[-1].h

print("Train Loss", cross_entropy(train_labels, y_hat))
print("Train Accuracy", accuracy(train_labels, y_hat))

ffn.forward_propogation(test_images)
y_hat_test = ffn.layers[-1].h

print("Test Loss", cross_entropy(test_labels, y_hat_test))
print("Test Accuracy", accuracy(test_labels, y_hat_test))
