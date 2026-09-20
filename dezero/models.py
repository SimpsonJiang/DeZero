from dezero import utils
import dezero.layers as L
import dezero.functions as F

class Model(L.Layer):
    def plot(self, *inputs, to_file = 'model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    def __init__(self, sizes:list, activation=F.sigmoid):
        super().__init__()
        self.act = activation
        self.layers = []

        last_size = sizes.pop(0)
        for i, size in enumerate(sizes):
            layer = L.Linear(last_size, size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
            last_size = size

            if i < len(sizes) - 1:
                act = L.Sigmoid()
                self.layers.append(act)
                setattr(self, 'act'+str(i), act)
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class Sequential(Model):
    def __init__(self, *args:L.Layer):
        super().__init__()

        self.layers = []
        self.layers_count = {}

        for layer in args:
            layer_name = type(layer).__name__
            if layer_name not in self.layers_count.keys():
                self.layers_count[layer_name] = 1
            else:
                self.layers_count[layer_name] += 1

            layer_reg_name = layer_name+str(self.layers_count[layer_name])
            setattr(self, layer_reg_name, layer)
            self.layers.append(layer)
    
    def __repr__(self):
        str = "Sequential(\n"
        for i, layer in enumerate(self.layers):
            str += f"({i}):"+layer.__repr__()+"\n"
        str += ")"
        return str


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def append(self, layer:L.Layer):
        layer_name = type(layer).__name__
        if layer_name not in self.layers_count.keys():
            self.layers_count[layer_name] = 1
        else:
            self.layers_count[layer_name] += 1
        
        layer_reg_name = layer_name+str(self.layers_count[layer_name])
        setattr(self, layer_reg_name, layer)
        self.layers.append(layer)