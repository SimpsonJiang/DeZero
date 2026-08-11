import numpy as np
from dezero.core import Variable, Function
import dezero.utils as utils

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
    
class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        return (1 - self.outputs[0]() * self.outputs[0]()) * gy

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
    
    def backward(self, gy):
        return reshape(gy, self.inputs[0].shape)

class Transpose(Function):
    def __init__(self, axes):
        self.axes = axes

    def forward(self, x):
        return x.transpose(self.axes)
    
    def backward(self, gy):
        if self.axes is not None:
            inv_axes = [0] * len(self.axes)
            for i, axis in enumerate(self.axes):
                inv_axes[axis] = i
            return transpose(gy, inv_axes)
        else:
            return transpose(gy)
        
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.sum(axis = self.axis, keepdims = self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.inputs[0].shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.inputs[0].shape)

class Broadcast_to(Function):
    def __init__(self, to_shape):
        self.to_shape = to_shape
    
    def forward(self, x):
        return np.broadcast_to(x, self.to_shape)

    def backward(self, gy):
        return sum_to(gy, self.inputs[0].shape)

class Sum_to(Function):
    def __init__(self, to_shape):
        self.to_shape = to_shape
    
    def forward(self, x):
        return utils.sum_to(x, self.to_shape)

    def backward(self, gy):
        return broadcast_to(gy, self.inputs[0].shape)


def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def reshape(x, shape):
    return Reshape(shape)(x)

def transpose(x, axes = None):
    return Transpose(axes)(x)

def sum(x, axis = None, keepdims = False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape ==  shape:
        return x
    return Broadcast_to(shape)(x)

def sum_to(x, shape):
    if x.shape ==  shape:
        return x
    return Sum_to(shape)(x)
