import numpy as np
class Variable:
    def __init__(self, data):
        data = np.array(data) if not isinstance(data,np.ndarray) else data
        self.data = data
        self.grad = np.array(None)

class Function:
    def __call__(self, x):
        assert isinstance(x, Variable), (
            f"x of Function must be a instance of Variable, got {type(x).__name__}"
        )
        self.x = x
        y = self.forward(x)
        return y
    
    def forward(self, x:Variable):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

if __name__ == "__main__":

    class Exp(Function):
        def forward(self, x):
            return Variable(np.exp(x.data))
        
        def backward(self, gy):
            return np.exp(self.x.data) * gy
    
    class Square(Function):
        def forward(self, x):
            return Variable(x.data ** 2)
        
        def backward(self, gy):
            return 2 * self.x.data * gy
    
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = 1
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    print(x.grad)

