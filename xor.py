"""
example of a function that can't be learned by a linear model is a XOR

"""

import numpy as np

from amrnet.train import train
from amrnet.nn import NeuralNet
from amrnet.layers import Linear, Tanh


inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# XNOR and XOR

targets = np.array([
    [1, 0],
    [0,1],
    [0,1],
    [1,0]
])

net = NeuralNet([
    Linear(2,2),
    Tanh(),
    Linear(2,2)
])

train(net,inputs, targets)

for x,y in zip(inputs, targets):
    
    predicted = net.forward(x)
    
    print(x, predicted, y)