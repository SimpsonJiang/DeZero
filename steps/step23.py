import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import * 
import numpy as np

def main():
    x1 = np.array(3)
    x2 = Variable(5)
    y = x1 / x2
    y.backward()

    print(y, x2.grad)

if __name__ == "__main__":
    main()