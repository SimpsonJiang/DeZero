from step01 import Variable
from step02 import Function, Square
import numpy as np

class Exp(Function):
    def forward(self, x):
        return Variable(np.exp(x.data))

if __name__ == "__main__":
    exp = Exp()
    square = Square()
    x = Variable(0.5)
    y = square(exp(square(x)))
    print(y.data)