""" 

A tensor is just an n dimensional array.

"""

# from numpy import ndarray as Tensor

from numpy import ndarray
import numpy as np
from numba import jit, vectorize, float64, njit
from typing_extensions import Self
from types import FunctionType




class Tensor(ndarray):
    
    
    @jit
    def __add__(self, b: Self) -> Self:
        
        return self + b
    
    
    @jit
    def __matmult__(self, b: Self) -> Self:
        return self @ b
    
    @jit
    def __mul__(self, b: Self) -> Self:
        return self * b
    
    @jit
    def __pow__(self, b: Self) -> Self:
        return self ** b
    
    @jit
    def __sub__(self, b: Self) -> Self:
        return self - b
    
    @jit
    def __truediv__(self, b: Self) -> Self:
        return self / b
    
    
    @staticmethod
    @jit
    def dot(a: Self, b: Self) -> Self:
        return np.dot(a, b)
    
    @staticmethod
    @vectorize([float64(float64, float64)])
    def maximum(a: Self, b: float) -> Self:
        return np.maximum(b, a)
    
    
    @staticmethod
    @vectorize([float64(float64)])
    def sqrt(a: Self) -> Self:
        return np.sqrt(a)
    
    
    @staticmethod
    @vectorize([float64(float64)])
    def log10(a: Self) -> Self:
        return np.log10(a)
    
    @staticmethod
    @vectorize([float64(float64)])
    def log(a: Self) -> Self:
        return np.log(a)
    
    
    @staticmethod
    @vectorize([float64(float64)])
    def tanh(a: Self) -> Self:
        return np.tanh(a)
    
    @staticmethod
    @vectorize([float64(float64)])
    def cosh(a: Self) -> Self:
        return np.cosh(a)
    
    
    
    @staticmethod
    @vectorize([float64(float64)])
    def exp(a: Self) -> Self:
        return np.exp(a)
    
    @staticmethod
    @njit
    def np_apply_along_axis(func1d: FunctionType, axis : int, arr: Self) -> Self:
        assert arr.ndim == 2
        assert axis in [0, 1]
        if axis == 0:
            result = np.empty(arr.shape[1])
            for i in range(len(result)):
                result[i] = func1d(arr[:, i])
        else:
            result = np.empty(arr.shape[0])
            for i in range(len(result)):
                result[i] = func1d(arr[i, :])
        return result
    
    
    
    @staticmethod
    @jit
    def max(a: Self, axis: int, keepdims: bool) -> Self:
        return np.max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    @vectorize([float64(float64)])
    def abs(a: Self) -> Self:
        return np.abs(a)
    
    @staticmethod
    @jit
    def clip(a: Self, min: float, max: float) -> Self:
        return np.clip(a, min, max)
    
    
    
    
    