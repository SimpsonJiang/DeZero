import numpy as np
import weakref
from contextlib import contextmanager

class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data: np.ndarray, name: str|None = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9) #换行后插入9个空格，即Variable(，这样分多行输出的时候就会对齐。
        return 'Variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            generations = np.array([f.generation for f in funcs])
            index = np.where(generations == generations.max())[0][-1]
            f = funcs.pop(index)

            gys = [y().grad for y in f.outputs]

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 由于y.grad以后不会再被用到，故释放其内存

            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,) #单个变量不是iterable的
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx.copy()
                else:
                    x.grad += gx.copy()
                if x.creator is not None and x.creator not in funcs:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   
            for output in outputs:
                output.set_creator(self) 
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

@contextmanager
def using_config(name, value):
    assert isinstance(name, str), print(f"Expected str, get {type(name)}.")
    assert hasattr(Config, name), print(f"Config does not have attributre \"{name}\".")
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

def square(x):
    return Square()(x)

def add(x1, x2):
    return Add()(x1, x2)

def main():
    # x1 = Variable(np.array(4))
    # x2 = Variable(np.array(5))
    # ys = add(square(x1), x2)
    # ys.backward()
    # print(ys.data)
    # print(x1.grad)
    x = Variable(np.array([[1,2,3],[1,2,3]]))
    print(x)
    print(len(x))

if __name__ == "__main__":
    main()