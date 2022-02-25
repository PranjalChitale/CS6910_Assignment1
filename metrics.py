import numpy as np
def accuracy(y_true, y_hat):
    '''
    Assumes y_true is one-hot encoded.
    Shape of y_hat & y_true = (batch_size, no_classes)
    '''
    true_label = np.argmax(y_true, axis = 1)
    pred_label = np.argmax(y_hat, axis = 1)
    return np.mean(true_label == pred_label) * 100
