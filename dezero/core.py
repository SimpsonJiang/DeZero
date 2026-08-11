import dezero
import numpy as np
import weakref
from contextlib import contextmanager
from typing import Optional

class Config:
    enable_backprop = True

class Variable:
    __array_priority__ = 1
    
    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = as_array(data)
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
    
    @property
    def T(self):
        return dezero.functions.transpose(self, None)
    
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
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self, axes = None):
        return dezero.functions.transpose(self, axes)
    
    def sum(self, axis = None, keepdims = False):
        return dezero.functions.sum(self, axis, keepdims)

    def backward(self, retain_grad = False, create_graph = False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            gys = [y().grad for y in f.outputs]

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 由于y.grad以后不会再被用到，故释放其内存
            
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,) #单个变量不是iterable的
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx 
                    if x.creator is not None:
                        add_func(x.creator)

class Function:
    def __call__(self, *inputs):
        inputs = [input if isinstance(input, Variable) else Variable(input) for input in inputs]
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

class Add(Function):
    def forward(self, x1, x2):
        y = x1 + x2
        return y
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x1, x2):
        y = x1 * x2
        return y
    def backward(self, gy):
        return gy * self.inputs[1], gy * self.inputs[0]

class Neg(Function):
    def forward(self, x1):
        return -x1
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x1, x2):
        return x1 - x2
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x1, x2):
        return x1 / x2
    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        return 1 / x1 * gy, -x0 / (x1 ** 2) * gy

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x1):
        return x1 ** self.c
    def backward(self, gy):
        x0 = self.inputs[0]
        return self.c * (x0 ** (self.c - 1)) * gy


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(x):
    if isinstance(x, np.ndarray):
        return Variable(x)
    elif isinstance(x, Variable):
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

def add(x1, x2):
    return Add()(x1, x2)

def mul(x1, x2):
    return Mul()(x1, x2)

def neg(x1):
    return Neg()(x1)

def sub(x1, x2):
    return Sub()(x1, x2)

def rsub(x1, x2):
    return Sub()(x2, x1)

def div(x1, x2):
    return Div()(x1, x2)

def rdiv(x1, x2):
    return Div()(x2, x1)

def pow(x1, c):
    return Pow(c)(x1)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow