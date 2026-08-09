import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import numpy as np

def f(x) -> Variable:
    return x ** 4 - 2 * x ** 2

def main():
    x = Variable(3)
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)


if __name__ == "__main__":
    main()