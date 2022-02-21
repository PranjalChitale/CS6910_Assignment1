import numpy as np
def accuracy(y_true, y_hat):
    true_label = np.argmax(y_true, axis = 1)
    pred_label = np.argmax(y_hat, axis = 1)
    return np.mean(true_label == pred_label) * 100
