import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(k, b, dim = 1, batch_size = 40, min_x = 0, max_x = 5, noise_sigma=0.3, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 5, batch_size).reshape(batch_size, dim, dim)
    y = k * x + b + noise_sigma * np.random.normal(0, size=x.shape)     

    # plt.plot(x, y, '.')
    # plt.plot(x, 3 * x + 2, 'r-')
    # plt.show()
    return x, y

def linear_function(x, W, b):
    y = x * W + b


def main():
    k_real = 2
    b_real = 3
    batch_size = 40
    min_x = 0
    max_x = 5
    gamma = 0.01
    dim = 1

    W = Variable(np.zeros(shape=(dim, dim)))
    b = Variable(np.zeros(shape=(1, dim)))

    best_loss = 10000
    history = np.zeros(shape=(3, 100))
    for i in range(100):
        data_x, data_y = load_dataset(k=k_real, b=b_real, dim=dim, batch_size=batch_size, min_x=min_x, max_x=max_x, seed = i)
        # print(data_x.shape, data_y.shape, W.shape, b.shape)
        y = F.matmul(data_x, W) + b
        loss = F.MSE_loss(data_y, y)
        history[0, i] = loss.data
        if loss.data < best_loss:
            best_loss = loss.data
            best_W = W
            best_b = b
        loss.backward()
        # print(f"y grad={y.grad}, W grad={W.grad.data}, b grad={b.grad.data} ")
        print(f"{i+1}th iter, loss={loss}")
        W.data = W.data - gamma * W.grad.data
        b.data = b.data - gamma * b.grad.data
        history[1, i] = W.data[0][0]
        history[2, i] = b.data[0][0]
        W.cleargrad()
    

    print(best_loss, W.data, b.data)
    with no_grad():
        x = np.array([np.linspace(0, 10, batch_size)]).reshape(batch_size, dim, dim)
        y = F.matmul(x, best_W) + best_b
    plt.plot(x[:,0,0], y.data[:,0,0], '.')
    plt.plot(x[:,0,0], k_real * x[:,0,0] + b_real, 'r-')
    plt.show()

    plt.plot(range(len(history[0])), history[0], 'y-', label="loss")
    plt.plot(range(len(history[1])), history[1], 'g-', label="W")
    plt.plot(range(len(history[2])), history[2], 'b-', label="b")
    plt.legend()
    plt.show()

def test():
    y = Variable(np.array([[1, 6], [5, 7]]))
    y_pred = np.array([[1, 7], [4, 7]])
    loss = F.MSE_loss(y_pred, y)
    loss.backward()
    print(loss)
    print(y.grad)

if __name__ == "__main__":
    main()