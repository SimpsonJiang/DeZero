from step01 import Variable
class Function:
    def __call__(self, input):
        assert isinstance(input, Variable), f"Input of Function must be Variable."
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

f = Square()
x = Variable(2)
y = f(x)
print(y.data)