#Imports
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split 


def flatten_images(dataset):
    """
    Converts each 28x28 image into a 784 dimensional input vector.
    Shape returned = (784, no_classes)
    """
    return np.array([dataset[i].flatten() for i in range(len(dataset))])

def one_hot_encoder(labels, no_classes):
    """
    Returns one hot representation of the label for each image.
    Shape returned = (no_images, no_classes)
    """
    temp = np.zeros((labels.shape[0], no_classes))
    for i in range(0,labels.shape[0]):
        temp[i][labels[i]] = 1
    return temp

#generate val set from train set and return val and updated train set.
def generate_val_set(train_images, train_labels, val_size):
    """
    Returns train_images, train_labels, val_images, val_labels.
    """
    train_images,val_images, train_labels, val_labels=train_test_split(train_images, train_labels,test_size=0.1,random_state=1)
    return train_images, train_labels, val_images, val_labels

def preprocess(train_images, train_labels, test_images, test_labels, val_size):
    """
    Preprocesses the data.
    1. Normalization.
    2. Flattens the images.
    3. One hot representation for the labels.
    """
    
    #Getting the number of classes.

    num_classes = np.unique(train_labels).shape[0]
    
    #Normalizing the data.
    #These are grayscale images so pixel values in range(0,255).
    #So we normalize them by dividing by 255 to get values in range(0,1).
    train_images=train_images / 255.0
    test_images = test_images / 255.0

    #Flatten out the images
    train_images = flatten_images(train_images)
    test_images = flatten_images(test_images)

    #Generate validation set here
    train_images, train_labels, val_images, val_labels = generate_val_set(train_images, train_labels, val_size)
    
    
    train_labels = one_hot_encoder(train_labels, num_classes)
    val_labels = one_hot_encoder(val_labels, num_classes)
    test_labels = one_hot_encoder(test_labels, num_classes)

    #return train_images, train_labels, val_images, val_labels, test_images, test_labels
    return train_images, train_labels, val_images, val_labels, test_images, test_labels




