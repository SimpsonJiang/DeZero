import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import *
import dezero.functions as F
import numpy as np
import matplotlib.pyplot as plt

def compute_derivatives(x_vals, n=3):
    x = Variable(x_vals.copy())
    y = F.sin(x)
    y.backward(create_graph=True)

    results = [y.data]
    for i in range(n):
        results.append(x.grad.data)
        gx = x.grad
        x.cleargrad()
        if i < n - 1:
            gx.backward(create_graph=True)
    return results

def main():
    x_vals = np.linspace(-5, 5, 200)
    y, dy1, dy2, dy3 = compute_derivatives(x_vals, n=3)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y, 'k-', label=r'$\sin(x)$', linewidth=1.5)
    ax.plot(x_vals, dy1, 'r--', label=r"$\sin'(x)=\cos(x)$", linewidth=1.5)
    ax.plot(x_vals, dy2, 'g-.', label=r"$\sin''(x)=-\sin(x)$", linewidth=1.5)
    ax.plot(x_vals, dy3, 'b:', label=r"$\sin'''(x)=-\cos(x)$", linewidth=2)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('sin(x) and its Derivatives')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('sin_derivatives.png', dpi=150)
    print("Plot saved to sin_derivatives.png")
    plt.show()

if __name__ == "__main__":
    main()
