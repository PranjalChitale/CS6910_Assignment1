#Question 1 
"Downloads the fashion-MNIST dataset and plot 1 sample image for each class in a grid."

#Imports
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb

#Getting the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Normalizing the data.
#These are grayscale images so pixel values in range(0,255).
#So we normalize them by dividing by 255 to get values in range(0,1).
train_images=train_images / 255.0
test_images = test_images / 255.0

#print("Shape of train image :- {}".format(train_images.shape)) 
#-> This shows we have 60000 train images having shape 28*28

#print("Shape of test image :- {}".format(test_images.shape))
#-> This shows we have 10000 test images having shape 28*28

#As, our labels contain class_id only, we use a dictionary to map it to textual labels.
class_dict = {0 : "T-shirt_top",
              1: "Trouser", 
              2: "Pullover", 
              3: "Dress", 
              4: "Coat", 
              5: "Sandal", 
              6: "Shirt", 
              7: "Sneaker", 
              8: "Bag", 
              9: "Ankle Boot"}

#This finds the unique class labels and returns corrresponding index of first occurence.
sample_image_class, sample_image_index = np.unique(train_labels, return_index=True)

#Using subplots to display the required grid.
fig = plt.figure(figsize=(10,5))
for i in range(0,len(sample_image_index)):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[sample_image_index[i]], cmap=plt.cm.binary)
    plt.xlabel(class_dict[sample_image_class[i]], fontsize=8, fontweight='bold')
wandb.log({"plot": plt})
plt.show()


