import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            f.x.grad = f.backward(f.y.grad)
            if f.x.creator is not None:
                funcs.append(f.x.creator)

class Function:
    def __call__(self, x) -> Variable:
        assert isinstance(x, Variable), (
            f"x of Function must be a instance of Variable, got {type(x).__name__}"
        )
        self.x = x
        y = self.forward(x)
        y.set_creator(self)
        self.y = y
        return y
    
    def forward(self, x:Variable) -> Variable:
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return Variable(as_array(x.data ** 2))
    
    def backward(self, gy):
        self.x.grad = gy * 2 * self.x.data
        return self.x.grad
    
class Exp(Function):
    def forward(self, x):
        return Variable(as_array(np.exp(x.data)))
    
    def backward(self, gy):
        self.x.grad = gy * np.exp(self.x.data)
        return self.x.grad

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def square(x:Variable):
    return Square()(x)

def exp(x:Variable):
    return Exp()(x)

if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)