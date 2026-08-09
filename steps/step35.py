import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np

def main():
    x = Variable(np.array(1))
    x.name = 'x'
    y = F.tanh(x)
    y.name = 'tanh'
    plot_dot_graph(y, './tanh.png')
    y.backward(create_graph=True)

    for i in range(3):
        print(f"{i+1}阶导数：{x.grad}")
        gx = x.grad
        gx.name = 'tanh' + '\'' * (i+1)
        plot_dot_graph(gx, f'./{gx.name}.png')
        x.cleargrad()
        gx.backward(create_graph=True)

if __name__ == "__main__":
    main()