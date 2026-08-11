import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np

def main():
    x1 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x2 = np.array([7, 8, 9])
    y = x1 ** x2
    y.backward()

    print(y)
    print(x1.grad)

if __name__ == "__main__":
    main()