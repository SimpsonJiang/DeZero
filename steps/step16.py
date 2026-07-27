import numpy as np
import psutil
import os

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
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            generations = np.array([f.generation for f in funcs])
            index = np.where(generations == generations.max())[0][-1]
            f = funcs.pop(index)

            xs, ys = f.inputs, f.outputs
            gys = [y.grad for y in ys]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,) #单个变量不是iterable的
            
            for x, gx in zip(xs, gxs):
                if x.grad is None:
                    x.grad = gx.copy()
                else:
                    x.grad += gx.copy()
                if x.creator is not None and x.creator not in funcs:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs) -> Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        self.generation = max([x.generation for x in inputs])

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

def main():
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024**2
    for i in range(10000):
        x = Variable(np.array(100))
        y = square(square(square(x)))
        y.backward()
    after = process.memory_info().rss / 1024**2
    print(f"Before {before} MiB \n After {after} MiB \n Used {after - before} MiB")


if __name__ == "__main__":
    main()