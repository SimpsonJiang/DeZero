from dezero import utils
import dezero.layers as L
import dezero.functions as F

class Model(L.Layer):
    def plot(self, *inputs, to_file = 'model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    def __init__(self, sizes, activation=F.sigmoid):
        super().__init__()
        self.act = activation
        self.layers = []

        last_size = sizes.pop(0)
        for i, size in enumerate(sizes):
            layer = L.Linear(last_size, size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
            last_size = size
        
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.act(l(x))
        return self.layers[-1](x)
