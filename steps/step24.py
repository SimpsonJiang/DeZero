import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import * 

def sphere(x1, x2):
    return x1 ** 2 + x2 ** 2

def matyas(x1, x2):
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2

def goldstein(x1, x2):
    return (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) * \
           (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))

def main():
    x1 = Variable(1)
    x2 = Variable(1)
    y = goldstein(x1, x2)
    y.backward()

    print(y, x1.grad, x2.grad)

if __name__ == "__main__":
    main()