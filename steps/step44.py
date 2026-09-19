import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.core import Variable, no_grad
import dezero.functions as F
import dezero.layers as L
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    l1 = L.Linear(1, 10)
    l2 = L.Linear(10, 1)

    def predict(x):
        x = F.sigmoid(l1(x))
        x = l2(x)
        return x

    lr = 0.2
    iters = 10000
    for i in range(iters):
        y_pred = predict(x)
        loss = F.MSE_loss(y_pred, y)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data

        if i % 100 == 0:
            print(f"loss={loss}")
    
    plt.plot(x, y, '.')
    with no_grad():
        x_data = np.linspace(0, 1, 100).reshape(100, 1)
        y_pred = predict(x_data)
    plt.plot(x_data, y_pred.data, 'r-')
    plt.show()


if __name__ == "__main__":
    main()