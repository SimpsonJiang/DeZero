import numpy as np
class Variable:
    def __init__(self, data):
        data = np.array(data) if not isinstance(data,np.ndarray) else data
        self.data = data
        self.grad = np.array(None)
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
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

if __name__ == "__main__":
    class Square(Function):
        def forward(self, x):
            return Variable(x.data ** 2)
        
        def backward(self, gy):
            self.x.grad = gy * 2 * self.x.data
            return self.x.grad
    
    class Exp(Function):
        def forward(self, x):
            return Variable(np.exp(x.data))
        
        def backward(self, gy):
            self.x.grad = gy * np.exp(self.x.data)
            return self.x.grad
    
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.8))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = 1.0
    y.backward()
    print(y.grad, b.grad, a.grad, x.grad)