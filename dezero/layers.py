import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter

class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype=self.dtype), name = "b")

    
    def _init_W(self):
        data_W = np.random.randn(self.in_size, self.out_size).astype(self.dtype) * np.sqrt(1 / self.in_size)
        self.W.data = data_W
    
    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        return F.linear(x, self.W, self.b)

class Sigmoid(Layer):
    def forward(self, x):
        return F.sigmoid(x)