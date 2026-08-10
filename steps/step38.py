import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np

def main():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.T
    y.backward()

    print(y, x.grad)

if __name__ == "__main__":
    main()