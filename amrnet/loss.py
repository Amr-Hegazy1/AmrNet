"""

A loss function measures how godd our predictions are.

Which is used to adjust network parameters

"""


import numpy as np

from tensor import Tensor



class Loss:
    
    
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        
        raise NotImplementedError
    

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        raise NotImplementedError
    
    
    
    
    
class TSE(Loss):
    
    """
    Total Squared Error
    
    """
    
    
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        
        return np.sum((predicted - actual) ** 2)
    

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        return 2 * (predicted - actual)
    
    
class MSE(Loss):
    
    """
    Mean Squared Error
    
    """
    
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        
        return Tensor.sum((predicted - actual) ** 2) / len(actual)
    

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        return 2 * (predicted - actual) / len(actual)
    
    
class MAE(Loss):
    
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return Tensor.sum(Tensor.abs(predicted - actual)) / len(actual)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        return (1 * ((predicted - actual) >= 0) - 1 * ((predicted - actual) < 0)) / len(actual)
    
    
class LogCosh(Loss):
    
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        
        return Tensor.sum(Tensor.log10(Tensor.cosh(predicted - actual)))
    

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        return Tensor.tanh(predicted - actual) / Tensor.log(10)
    
    
class Huber(Loss):
    
    def __init__(self, delta: float = 1.0) -> None:
        
        self.delta = delta
        
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        
        return Tensor.sum(np.where(Tensor.abs(predicted - actual) < self.delta, 0.5 * (predicted - actual) ** 2, self.delta * Tensor.abs(predicted - actual) - 0.5 * self.delta ** 2))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        return Tensor.where(Tensor.abs(predicted - actual) < self.delta, predicted - actual, self.delta * Tensor.sign(predicted - actual))
    
    
class CrossEntropy(Loss):
    
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        
        predicted_clipped = Tensor.clip(predicted, 1e-7, 1 - 1e-7)
        
        return -Tensor.sum(actual * np.log(predicted_clipped) + (1 - actual) * Tensor.log(1 - predicted_clipped))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        
        predicted_clipped = Tensor.clip(predicted, 1e-7, 1 - 1e-7)
        
        return -(actual / predicted_clipped) + (1 - actual) / (1 - predicted_clipped)