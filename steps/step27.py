import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import numpy as np
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

def main():
    x = Variable(np.pi / 4)
    y = my_sin(x)
    y.backward()

    print(y, x.grad)
    plot_dot_graph(y, "graph.png")

if __name__ == "__main__":
    main()