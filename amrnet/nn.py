"""
A neural net is basically a collection of layers
"""


from tensor import Tensor

from layers import Layer


class NeuralNet:
    
    
    def __init__(self,layers) -> None:
        self.layers = layers
        
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        for layer in self.layers:
            
            inputs = layer.forward(inputs)
            
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        
        for layer in reversed(self.layers):
            
            grad = layer.backward(grad)
            
        return grad
    
    
    def params_and_grads(self):
        
        for layer in self.layers:
            
            for name, param in layer.params.items():
                
                grad = layer.grads[name]
                
                yield param, grad
        