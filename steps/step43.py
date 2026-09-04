import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(dim = 1, batch_size = 40, min_x = 0, max_x = 1, noise_sigma=0.3, seed=42):
    np.random.seed(seed)
    x = max_x * np.random.random(batch_size).reshape(batch_size, dim)
    y = np.sin(2*np.pi*x) + noise_sigma * np.random.normal(0, size=x.shape)     

    # plt.plot(x, y, '.')
    # plt.plot(x, 3 * x + 2, 'r-')
    # plt.show()
    return x, y

def linear_function(x, W, b):
    y = x * W + b


def main():
    batch_size = 100
    min_x = 0
    max_x = 1
    gamma = 0.15
    input = 1
    hidden = 20
    output = 1
    seed = 100
    np.random.seed(seed)

    W1 = Variable(np.random.randn(input, hidden))
    b1 = Variable(np.zeros((input, hidden)))
    W2 = Variable(np.random.randn(hidden, output))
    b2 = Variable(np.zeros((1, output)))

    def model(x):
        x = F.sigmoid(F.linear(x, W1, b1))
        y = F.linear(x, W2, b2)
        return y


    best_loss = 10000
    history = np.zeros(shape=(3, 10000))
    for i in range(10000):

        data_x, data_y = load_dataset(dim=input, batch_size=batch_size, min_x=min_x, max_x=max_x, seed = i)
        # print(data_x.shape, data_y.shape, W.shape, b.shape)
        y = model(data_x)
        loss = F.MSE_loss(data_y, y)

        history[0, i] = loss.data
        if loss.data < best_loss:
            best_loss = loss.data
            best_W1 = W1
            best_b1 = b1
            best_W2 = W2
            best_b2 = b2

        loss.backward()
        #print(f"W1_grad={W1.grad.data}, b1_grad={b1.grad.data}, W2_grad={W2.grad.data}, b2_grad={b2.grad.data} ")
        print(f"{i+1}th iter, loss={loss}")

        W1.data -= gamma * W1.grad.data
        b1.data -= gamma * b1.grad.data
        W2.data -= gamma * W2.grad.data
        b2.data -= gamma * b2.grad.data

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()
    

    print(best_loss)
    with no_grad():
        x = np.array([np.linspace(0, max_x, 200)]).reshape(200, input)
        W1 = best_W1
        W2 = best_W2
        b1 = best_b1
        b2 = best_b2
        y = model(x)
    plt.plot(x[:,0], y.data[:,0], '.')
    plt.plot(data_x[:,0], data_y[:,0], '.')
    plt.plot(x[:,0], np.sin(2*np.pi*x[:,0]), 'r-')
    plt.show()

    plt.plot(range(len(history[0])), history[0], 'y-', label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()