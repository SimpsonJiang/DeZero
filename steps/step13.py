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
            xs, ys = f.inputs, f.outputs
            gys = [y.grad for y in ys]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,) #单个变量不是iterable的
            for x, gx in zip(xs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx
                if x.creator is not None:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs) -> Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, *xs):
        raise NotImplementedError()
    
    def backward(self, *gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        gx = gy * 2 * self.inputs[0].data
        return gx

class Add(Function):
    def forward(self, x1, x2):
        y = x1 + x2
        return y
    def backward(self, gy):
        return gy, gy

def square(x):
    return Square()(x)

def add(x1, x2):
    return Add()(x1, x2)

if __name__ == "__main__":
    x1 = Variable(np.array(3))
    x2 = Variable(np.array(5))
    ys = add(x1, x1)
    ys.backward()
    print(ys.data)
    print(x1.grad)