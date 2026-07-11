import numpy as np
class Variable:
    def __init__(self, data):
        data = np.array(data) if not isinstance(data,np.ndarray) else data
        self.data = data

if __name__ == "__main__":
    x = Variable([1,2])
    y = Variable(np.array(2))
    print(type(x.data))
    print(type(y.data))