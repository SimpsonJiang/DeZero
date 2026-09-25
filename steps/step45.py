import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Variable, no_grad
import dezero.functions as F
import dezero.layers as L
import dezero.models as M
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(M.Model):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(in_size, hidden_size)
        self.l2 = L.Linear(hidden_size, out_size)
        self.sigmoid = F.sigmoid
    
    def forward(self, x):
        x = self.sigmoid(self.l1(x))
        x = self.l2(x)
        return x

def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 * np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    model = M.Sequential(
        L.Linear(1, 10),
        L.Sigmoid(),
        L.Linear(10, 1)
    )
    print(model)
    model.plot(x)

    lr = 0.2
    epochs = 10000
    batch_size = 100
    data_size = len(x)
    for epoch in range(epochs):
        indices = np.random.permutation(data_size)
        for start in range(0, data_size, batch_size):
            batch_x = x[indices[start:start+batch_size]]
            batch_y = y[indices[start:start+batch_size]]
            y_pred = model(batch_x)
            loss = F.MSE_loss(y_pred, batch_y)
            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad.data
            model.cleargrads()
        if epoch % 100 == 0:
            print(f"loss={loss}")
    
    plt.plot(x, y, '.')
    with no_grad():
        x_data = np.linspace(0, 1, 100).reshape(100, 1)
        y_pred = model(x_data)
    plt.plot(x_data, y_pred.data, 'r-')
    plt.show()

    


if __name__ == "__main__":
    main()