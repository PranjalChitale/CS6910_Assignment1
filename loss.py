import numpy as np
def cross_entropy(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    true_label = np.argmax(y_true, axis = 1)
    pred_label = np.argmax(y_hat, axis = 1)
    true_label_col = np.array(range(0,true_label.shape[0]))
    temp = y_hat[true_label_col, true_label]
    loss =  -np.log(temp)
    return np.mean(loss)

def squared_error(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    loss = np.mean(np.square(y_hat - y_true))
    return loss

