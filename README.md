# CS6910 Assignment1 : February 2022

Implementation of a Feed Forward Network and various Optimization Algorithms for training the same.
The implementation is from scratch and is based on numpy.
The task to be performed is multi-class classification.
Fashion MNIST dataset is considered in this implementation.
However, with minor changes the code can be adapted to other datasets too.

## How to define the Neural Network for classification ?
```
model = FFN(input_dim = 784, num_classes = 10, optimizer=config.optimizer, weight_decay=config.weight_decay, loss_fn= config.loss_fn)
```

## How to add a Hidden Layer to the network.
```
model.add_layer(config.hidden_layer_size, weight_init_method=config.weight_init, activation_func=config.activation_func)
```

## How to add a New Optimizer function.

Write code for the new optimizer in 'optimizers.py', and also add condition to invoke the optimizer in the if condition present in train function of 'nn.py'
```
# optimizers.py

def new_optimizer(ffn, train_images, train_labels, val = None, epochs =1 , lr = 0.001, batch_size = 1, opt_specific_params):
    #Write code for new optimizer here

#nn.py -> train function
elif self.optimizer == "new_optimizer":
    new_optimizer(self, train_images, train_labels, val = val, epochs = epochs, lr = learning_rate, batch_size = batch_size, opt_specific_params)   

#Also provide the default value of opt_specific_params in train function.
This is required as we are invoking the train function which is in turn invoking the optimizer.
Therefore, we require a default value of the parameters to pass to the optimizer.

#Done!
```
## Loss Functions supported
cross_entropy : Cross Entropy
squared_error : Squared error.

## Train a network
```
#Define the network.
model = FFN(input_dim = 784, num_classes = 10, optimizer="vgd",weight_decay=0.001, loss_fn="cross_entropy")

#Add the requisite hidden layers.

model.add_layer(config.hidden_layer_size, weight_init_method=config.weight_init, activation_func=config.activation_func)

#Add last layer
model.add_layer(10,weight_init_method="xavier_init",activation_func="linear")

#Train the model
model.train(train_images, train_labels, val = (val_images, val_labels), learning_rate= 0.0001, batch_size=128,epochs=10)

```

## Test the Model :

```
train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

#Loads our trained model
with open('model_pkl' , 'rb') as f:
    ffn = pickle.load(f)

#Uses it to get predictions on the test set.
test_pred = ffn.get_prediction(test_images)

```

## Use WANDB sweep

```
# In wandb_sweep.py

#First Set your project name and username
wandb.init(project=project_name, entity=entity)

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
wandb.agent(sweep_id, project=project_name, function=train_wandb)

#Change the config_sweep as per the need, to sweeo using different strategy.

```
