import numpy as np

def random_init(num_neurons, prev_neurons):
    return np.random.randn(prev_neurons, num_neurons)

