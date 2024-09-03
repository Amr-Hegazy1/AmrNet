"""
Layers make up neural nets

Each layer needs to do a forward pass

then propagate its gradients backwards

"""

import numpy as np

from tensor import Tensor


class Layer:
    
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}
    
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the output coressponding to these inputs
        """
        
        raise NotImplementedError
    
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Back propagate this gradient through the layer
        """
        
        raise NotImplementedError
    
    

class Linear(Layer):
    
    """
    output = inputs @ w + b
    
    """
    
    
    def __init__(self, input_size: int, output_size: int) -> None:
        
        super().__init__()
        
        # inputs are (batch_size,input_size)
        
        # outputs are (batch_size,output_size)
        
        self.params['w'] = np.random.randn(input_size,output_size)
        
        self.params['b'] = np.random.randn(output_size)
        
        
    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = inputs @ w + b
        
        """
        
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    
    def backward(self, grad: Tensor) -> Tensor:
        
        
        self.grads['b'] = np.sum(grad,axis=0)
        
        self.grads['w'] = self.inputs.T @ grad
        
        return grad @  self.params['w'].T
    


 
    
class Activation(Layer):
    
    """
    Applies a function elementwise to its inputs
    """
    
    def __init__(self, fn , fn_prime, *args) -> None:
        super().__init__()
        
        self.fn = fn
        
        self.fn_prime = fn_prime
        
        self.args = args
        
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        self.inputs = inputs
        
        return self.fn(inputs,*self.args)
    
    def backward(self, grad: Tensor) -> Tensor:
        
        
        return self.fn_prime(self.inputs,*self.args) * grad
        
        
        
        
        
def tanh(x: Tensor) -> Tensor:
    
    return Tensor.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    
    y = tanh(x)
    
    return 1- y ** 2



class Tanh(Activation):
    
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)
        

def relu(x):
    return Tensor.maximum(0,x)     

def relu_prime(x):
    
    return 1 * (x > 0)
      
        
class RelU(Activation):
    
    def __init__(self) -> None:
        
        super().__init__(relu, relu_prime)
        
        
def leaky_relu(x,negative_slope):
    
    return Tensor.maximum(x * negative_slope,x)

def leaky_relu_prime(x,negative_slope):
    
    return 1 * (x > 0) + negative_slope * (x <= 0)
    
        
class LeakyRelU(Activation):
    
    def __init__(self, negative_slope=0.01) -> None:
        super().__init__(leaky_relu, leaky_relu_prime,negative_slope)
        
        
def sigmoid(x):
        
        return 1 / (1 + Tensor.exp(-x))
    
def sigmoid_prime(x):
    
    return sigmoid(x) * (1 - sigmoid(x))

class Sigmoid(Activation):
    
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)
        

def softmax(x):
        
        exps = Tensor.exp(x - Tensor.max(x,axis=1,keepdims=True))
        
        return exps / Tensor.sum(exps,axis=1,keepdims=True)
    
def softmax_prime(x):
    
    return softmax(x) * (1 - softmax(x))

class Softmax(Activation):
    
    def __init__(self) -> None:
        super().__init__(softmax, softmax_prime)
        
        
class Dropout(Layer):
    
    def __init__(self, p: float = 0.5) -> None:
        
        super().__init__()
        
        self.p = p
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        if self.p == 0:
            return inputs
        
        self.mask = np.random.binomial(1,1-self.p,size=inputs.shape) / (1-self.p)
        
        return inputs * self.mask
    
    def backward(self, grad: Tensor) -> Tensor:
            
            return grad * self.mask
        

class RNN(Layer):
        
        def __init__(self, input_size: int, hidden_size: int) -> None:
            
            super().__init__()
            
            self.input_size = input_size
            
            self.hidden_size = hidden_size
            
            self.params['w'] = np.random.randn(input_size + hidden_size,hidden_size)
            
            self.params['b'] = np.random.randn(hidden_size)
            
        def forward(self, inputs: Tensor) -> Tensor:
            
            self.inputs = inputs
            
            self.h = Tensor.zeros((inputs.shape[0],self.hidden_size))
            
            self.outputs = []
            
            for x in inputs:
                
                self.h = Tensor.tanh(np.dot(np.hstack((x,self.h)),self.params['w']) + self.params['b'])
                
                self.outputs.append(self.h)
                
            return np.array(self.outputs)
        
        def backward(self, grad: Tensor) -> Tensor:
            
            grad = grad.copy()
            
            self.grads['b'] = np.sum(grad,axis=0)
            
            self.grads['w'] = np.zeros_like(self.params['w'])
            
            for i in reversed(range(len(self.inputs))):
                
                x = self.inputs[i]
                
                self.grads['w'] += Tensor.dot(np.hstack((x,self.h)).T,grad[i])
                
                grad[i] = Tensor.dot(grad[i],self.params['w'].T)
                
                self.h = self.outputs[i]
                
            return grad
        

class LSTM(Layer):
    
    def __init__(self, input_size: int, hidden_size: int) -> None:
        
        super().__init__()
        
        self.input_size = input_size
        
        self.hidden_size = hidden_size
        
        self.params['w'] = np.random.randn(input_size + hidden_size,hidden_size * 4)
        
        self.params['b'] = np.random.randn(hidden_size * 4)
        
    def forward(self, inputs: Tensor) -> Tensor:
        
        self.inputs = inputs
        
        self.h = np.zeros((inputs.shape[0],self.hidden_size))
        
        self.c = np.zeros((inputs.shape[0],self.hidden_size))
        
        self.outputs = []
        
        for x in inputs:
            
            a = Tensor.dot(np.hstack((x,self.h)),self.params['w']) + self.params['b']
            
            ai,af,ao,ag = np.split(a,4,axis=1)
            
            i = sigmoid(ai)
            
            f = sigmoid(af)
            
            o = sigmoid(ao)
            
            g = Tensor.tanh(ag)
            
            self.c = self.c * f + g * i
            
            self.h = Tensor.tanh(self.c) * o
            
            self.outputs.append(self.h)
            
        return np.array(self.outputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        
        grad = grad.copy()
        
        self.grads['b'] = np.sum(grad,axis=0)
        
        self.grads['w'] = np.zeros_like(self.params['w'])
        
        for i in reversed(range(len(self.inputs))):
            
            x = self.inputs[i]
            
            self.grads['w'] += Tensor.dot(np.hstack((x,self.h)).T,grad[i])
            
            grad_i = Tensor.dot(grad[i],self.params['w'].T)
            
            grad_f = grad_o = grad_g = np.zeros_like(grad_i)
            
            grad_c = grad_c * self.f[i] + self.o[i] * (1 - np.tanh(self.c[i]) ** 2) * grad_i
            
            grad[i] = Tensor.dot(grad_c,self.params['w'].T)
            
            self.h = self.outputs[i]
            
        return grad
    
