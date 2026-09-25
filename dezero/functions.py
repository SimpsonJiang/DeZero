import math
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

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        self.x_shape = x.shape
        return x[self.slices]
    
    def backward(self, gy):
        f = GetItemGrad(self.slices, self.x_shape)
        return f(gy)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, x):
        y = np.zeros(self.in_shape)
        np.add.at(y, self.slices, x)
        return y
    
    def backward(self, gy):
        return get_item(gy, self.slices)

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

class Mean(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        self.x_size = x.size
        return x.mean(axis = self.axis, keepdims = self.keepdims)

    def backward(self, gy):
        if self.axis is not None:
            gx = utils.reshape_sum_backward(gy, self.inputs[0].shape, self.axis, self.keepdims)  / self.x_shape[self.axis]
        else:
            gx = utils.reshape_sum_backward(gy, self.inputs[0].shape, self.axis, self.keepdims)  / self.x_size
        return broadcast_to(gx, self.inputs[0].shape)

class MatMul(Function):
    def forward(self, x1, x2):
        return np.matmul(x1, x2)
    
    def backward(self, gy):
        x1, x2 = self.inputs
        x1_T_shape = list(range(len(x1.shape) - 2)) + [len(x1.shape) - 1, len(x1.shape) - 2]
        x2_T_shape = list(range(len(x2.shape) - 2)) + [len(x2.shape) - 1, len(x2.shape) - 2]
        x1_T = x1.transpose(x1_T_shape)
        x2_T = x2.transpose(x2_T_shape)
        gx1, gx2 =  matmul(gy, x2_T), matmul(x1_T, gy)
        return sum_to(gx1, x1.shape), sum_to(gx2, x2.shape)

class Linear(Function):
    def forward(self, x, W, b):
        return np.matmul(x, W) + b
    
    def backward(self, gy):
        x, W, b = self.inputs
        x_T_shape = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
        x_T = x.transpose(x_T_shape)
        W_T_shape = list(range(len(W.shape) - 2)) + [len(W.shape) - 1, len(W.shape) - 2]
        W_T = W.transpose(W_T_shape)
        gx = matmul(gy, W_T)
        gW = matmul(x_T, gy)
        return sum_to(gx, x.shape), sum_to(gW, W.shape), sum_to(gy, b.shape)

class Sigmoid(Function):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, gy):
        return gy * self.outputs[0]() * (1 - self.outputs[0]())

class MeanSquareError(Function):
    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f"MSE needs two arraies with same shape, got {x1.shape} and {x2.shape}."
        self.x1 = x1
        self.x2 = x2
        return np.mean((x1 - x2) ** 2)

    def backward(self, gy):
        gx1 = gy / self.x1.size * 2 * (self.inputs[0] - self.inputs[1])
        gx2 = gy / self.x1.size * 2 * (self.inputs[1] - self.inputs[0])
        return gx1, gx2

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

def get_item(x, slice):
    return GetItem(slice)(x)

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

def mean(x, axis = None, keepdims = False):
    return Mean(axis, keepdims)(x)

def matmul(x1, x2):
    return MatMul()(x1, x2)

def linear(x, W, b):
    return Linear()(x, W, b)

def sigmoid(x):
    return Sigmoid()(x)

def MSE_loss(x1, x2):
    return MeanSquareError()(x1, x2)