from step01 import Variable
from step02 import Function

def numerical_diff(f, x, eps=1e-4):
    assert isinstance(f, Function), (
        f"f must be a instance of Function, got {type(f).__name__}"
    )
    assert isinstance(x, Variable), (
        f"x must be a instance of Variable, got {type(x).__name__}"
    )

    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

if __name__ == "__main__":
    from step02 import Square
    from step03 import Exp
    class Cube(Function):
        def forward(self, x):
            return Variable(x.data ** 3)
    
    
    cube = Cube()
    square = Square()
    exp = Exp()
    class Comp(Function): # e^{x^2}
        def forward(self, x):
            return exp(square(x))
    comp = Comp()

    x = Variable(1)
    f = comp
    diff = numerical_diff(f, x)
    print(f"x = {x.data}, y = {f(x).data}, diff = {diff}")