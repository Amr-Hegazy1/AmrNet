"""

A loss function measures how godd our predictions are.

Which is used to adjust network parameters

"""


import numpy as np

from amrnet.tensor import Tensor



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
    
    