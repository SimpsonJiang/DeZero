import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np

def main():
    x1 = Variable(np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]]))
    x2 = Variable(np.array([[2, 3], [4, 5]]))
    y = F.matmul(x1, x2)
    y.backward()

    print(y.grad)
    print(x1.grad)
    print(x2.grad)

if __name__ == "__main__":
    main()