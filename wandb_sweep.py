from keras.datasets import fashion_mnist
import numpy as np
from preprocess_f_mnist import preprocess
from nn import FFN
from loss import *
from metrics import accuracy
import wandb


def train_wandb(config = None):
    #Getting the data
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    #Defining the validation set size and preprocessing.
    val_size = 0.10 
    train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels, val_size)

    run = wandb.init(config=config, resume=True)
    config = wandb.config

    name = f'hl_{config.hidden_layers}_bs_{config.batch_size}_acf_{config.activation_func}_lr_{config.learning_rate}_opt_{config.optimizer}_w_init_{config.weight_init}_wdecay_{config.weight_decay}'
    wandb.run.name = name
    wandb.run.save()

    model = FFN(input_dim = 784, num_classes = 10, optimizer=config.optimizer, weight_decay=config.weight_decay, loss_fn="cross_entropy")

    for i in range(config.hidden_layers) :
        model.add_layer(config.hidden_layer_size, weight_init_method=config.weight_init, activation_func=config.activation_func)

    model.add_layer(10, weight_init_method = config.weight_init, activation_func = "linear")

    model.train(train_images, train_labels, val = (val_images, val_labels), learning_rate = config.learning_rate, batch_size=config.batch_size, epochs=config.epochs)

project_name = '' #Add project name here
entity = '' #Add username here
wandb.init(project=project_name, entity=entity)

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
              },
    'parameters': {
        'epochs': {
            'values': [5,10]
        },
        'hidden_layers': {
            'values': [3,4,5]
        },
        'hidden_layer_size' : {
            'values' : [32,64,128,256]
        },
        'learning_rate': {
            'values': [0.001,0.0001]
        },
        'optimizer': {
            'values': ["vgd","mgd","nag", "rmsprop", "adam","nadam"]
        },
        'batch_size': {
            'values': [16,32,64,128]
        },
        'weight_init': {
            'values': ["random_init", "xavier_init", "kaiming_init"]
        },
        'activation_func': {
            'values': ["sigmoid","tanh","relu"]
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        }
    }
}

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

wandb.agent(sweep_id, project=project_name, function=train_wandb)