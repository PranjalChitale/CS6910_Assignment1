from keras.datasets import fashion_mnist
import numpy as np
from preprocess_f_mnist import preprocess
from nn import FFN
from loss import *
from metrics import accuracy
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
import pandas as pd
import plotly
import wandb

#Getting the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Defining the validation set size and preprocessing.
val_size = 0.10 

train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

#Loads our trained model
with open('model_pkl' , 'rb') as f:
    ffn = pickle.load(f)

#Uses it to get predictions on the test set.
test_pred = ffn.get_prediction(test_images)

true_label = np.argmax(test_labels, axis = 1)
pred_label = np.argmax(test_pred, axis = 1)


cm_plot_labels = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
			   'Ankle boot']

cm=metrics.confusion_matrix(y_true=true_label,y_pred=pred_label)

df_cm = pd.DataFrame(cm, index=[i for i in cm_plot_labels],
						 columns=[i for i in cm_plot_labels])


fig = px.imshow(df_cm,
                x=[i for i in cm_plot_labels],
                y=[i for i in cm_plot_labels],
                labels=dict(x="True Class", y="Predicted Class"),
                text_auto=True
               )

fig.update_layout(title_text='Test Accuracy = {}'.format(accuracy(test_labels, test_pred)))

fig.update_xaxes(side="top")

wandb.log({"plot":fig})

fig.show()

