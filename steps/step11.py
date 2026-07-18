import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

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
            f.input.grad = f.backward(f.output.grad)
            if f.input.creator is not None:
                funcs.append(f.input.creator)

class Function:
    def __call__(self, inputs) -> Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs):
        y = xs[0] + xs[1]
        return (y,) # 这样Return的就会是tuple，如果不加,的话，括号就被识别为运算中的括号，返回y本身的值

if __name__ == "__main__":
    add = Add()
    xs = [Variable(np.array(1)), Variable(np.array(2))]
    ys = add(xs)
    print(ys[0].data)