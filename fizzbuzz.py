import numpy as np

from amrnet.train import train
from amrnet.nn import NeuralNet
from amrnet.layers import Linear, Tanh

from amrnet.optimizers import SGD, Momentum

from amrnet.loss import TSE, MSE, LogCosh, MAE




def fizz_buzz_encode(x: int) -> [int]:
    
    if x % 15 == 0:
        return [0, 0, 0, 1]
    
    elif x % 5 == 0:
        
        return [0, 0, 1, 0]
    
    elif x % 3 == 0:
        return [0, 1, 0, 0]

    else:
        
        return [1, 0, 0, 0]
    
    
def binary_encode(x: int) -> [int]:
    
    """
    10 digit binary encodeing of x
    """
    
    return [x >> i & 1 for i in range(10)]



inputs = np.array([
    binary_encode(x)
    for x in range(101,1024)
    
])

targets = np.array([
    
    fizz_buzz_encode(x) 
    
    for x in range(101,1024)
    
])


net = NeuralNet([
    
    Linear(10,50),
    Tanh(),
    Linear(50,4)
    
])

train(net,inputs, targets,optimizer=SGD(0.001),loss=MAE())

for x in range(1, 101):
    
    predicted = net.forward(binary_encode(x))
    
    predicted_idx = np.argmax(predicted)
    
    
    actual_idx = np.argmax(fizz_buzz_encode(x))
    
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    
    print(x, labels[predicted_idx],labels[actual_idx])