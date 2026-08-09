import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        return cos(self.inputs[0]) * gy

class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        return -sin(self.inputs[0]) * gy

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)
