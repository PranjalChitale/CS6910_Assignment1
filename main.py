#Imports
from keras.datasets import fashion_mnist
import numpy as np
from preprocess_f_mnist import preprocess

#Getting the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

val_size = 0.10 

#train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

train_images, train_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)