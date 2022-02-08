def cross_entropy(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    true_label = np.argmax(y_true)
    pred_label = np.argmax(y_hat)

    loss =  -np.log(y_hat[true_label])
    return loss

def squared_error(y_true, y_hat):
    """
    Assumes, y_true and y_hat are one-hot vectors.
    """
    loss = np.mean(np.square(y_hat - y_true))
    return loss
