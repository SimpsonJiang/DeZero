import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x0, x1, a=1, b=100):
    return b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2

def gradient_descent(func, *xs, gamma = 0.0015, iter = 10e4):
    history = [(float(xs[0].data), float(xs[1].data))]
    grad_norms = []
    y = func(*xs)
    while len(history) < iter:
        y.backward()
        grad_norm = math.sqrt(sum(float(x.grad)**2 for x in xs))
        grad_norms.append(grad_norm)
        for i in range(len(xs)):
            xs[i].data = xs[i].data - gamma / np.sqrt(np.sqrt(len(history))) * xs[i].grad
            xs[i].cleargrad()
        y = func(*xs)
        history.append((float(xs[0].data), float(xs[1].data)))

    return *xs, y, history, grad_norms

def plot_rosenbrock(history, grad_norms, a=1, b=100):
    history = np.array(history)

    x0_grid = np.linspace(-0.5, 2.0, 400)
    x1_grid = np.linspace(-0.5, 3.0, 400)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    Z = b * (X1 - X0 ** 2) ** 2 + (a - X0) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    contour = ax1.contour(X0, X1, Z, levels=np.logspace(-1, 3, 15), cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)

    ax1.plot(history[:, 0], history[:, 1], 'r.-', markersize=4, linewidth=0.5, label='Gradient Descent')
    ax1.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(history[-1, 0], history[-1, 1], 'b*', markersize=15, label='End')

    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_title('Gradient Descent on Rosenbrock Function')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(grad_norms, 'b-', linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm over Iterations')
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=150)
    print("Plot saved to gradient_descent.png")
    plt.show()

def main():
    x0 = Variable(0.0)
    x1 = Variable(2.0)
    optimal_x0, optimal_x1, optimal_y, history, grad_norms = gradient_descent(rosenbrock, x0, x1)

    print(f"Optimal y={optimal_y.data} at ({optimal_x0.data}, {optimal_x1.data})")
    plot_rosenbrock(history, grad_norms)

if __name__ == "__main__":
    main()
