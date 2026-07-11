from step01 import Variable
class Function:
    def __call__(self, x):
        assert isinstance(x, Variable), (
            f"x of Function must be a instance of Variable, got {type(x).__name__}"
        )
        output = self.forward(x)
        return output
    
    def forward(self, x:Variable):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return Variable(x.data ** 2)

if __name__ == "__main__":
    f = Square()
    x = Variable(3)
    y = f(x)
    print(y.data)