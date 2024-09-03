"""
An optimizer is used to adjust the network parameters based on gradients computed from back propagtion
"""
import numpy as np

from tensor import Tensor

from nn import NeuralNet

class Optimizer:
    
    def step(self, net: NeuralNet) -> None:
        
        raise NotImplementedError
    
    
class SGD(Optimizer):
    
    def __init__(self, lr: float = 0.01) -> None:
        
        self.lr = lr
        
        
    def step(self, net: NeuralNet) -> None:
        
        for param, grad in net.params_and_grads():
            
            param -= self.lr * grad
            
            
class Adam(Optimizer):
        
        def __init__(self,
                    lr: float = 0.001,
                    beta_1: float = 0.9,
                    beta_2: float = 0.999,
                    epsilon: float = 1e-7) -> None:
            
            self.lr = lr
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon
            
            self.m = 0
            self.v = 0
            self.k = 0
            
            
        def step(self, net: NeuralNet) -> None:
            
            self.k += 1
            
            for param, grad in net.params_and_grads():
                
                self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
                self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad ** 2
                
                m_hat = self.m / (1 - self.beta_1 ** self.k)
                v_hat = self.v / (1 - self.beta_2 ** self.k)
                
                param -= self.lr * m_hat / (Tensor.sqrt(v_hat) + self.epsilon)
            
            
            
                
           