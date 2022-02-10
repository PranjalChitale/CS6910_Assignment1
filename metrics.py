import numpy as np
def accuracy(y_true, y_hat):
    true_label = np.argmax(y_true, axis = 1)
    pred_label = np.argmax(y_hat, axis = 1)
    acc = np.equal(true_label, pred_label)
    ctr = 0
    for i in acc:
        if i == True:
            ctr+=1
    return(ctr/acc.shape[0]*100)
