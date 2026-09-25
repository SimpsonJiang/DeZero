import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = list(params)
        self.hooks = []
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for param in self.params:
            param.cleargrad()
    
    def apply_hook(self, f):
        self.hooks.append(f)
        f(self.params)

class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
    
    def step(self):
        for param in self.params:
            param_id = id(param)
            if param_id not in self.vs.keys():
                self.vs[param_id] = np.zeros_like(param.grad.data)
            self.vs[param_id] = self.momentum * self.vs[param_id] - self.lr * param.grad.data
            param.data += self.vs[param_id]

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=[0.9, 0.999], eps=1e-8):
        super().__init__(params=params)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.ms = {}
        self.vs = {}
        self.t = 0

    def step(self):
        self.t += 1
        for param in self.params:
            param_id = id(param)
            if param_id not in self.ms:
                self.ms[param_id] = np.zeros_like(param.grad.data)
            if param_id not in self.vs:
                self.vs[param_id] = np.zeros_like(param.grad.data)
            
            self.ms[param_id] = self.betas[0] * self.ms[param_id] + (1 - self.betas[0]) * param.grad.data
            self.vs[param_id] = self.betas[1] * self.vs[param_id] + (1 - self.betas[1]) * param.grad.data ** 2
            self.hat_m = self.ms[param_id] / (1 - self.betas[0] ** self.t)
            self.hat_v = self.vs[param_id] / (1 - self.betas[1] ** self.t)
            param.data -= self.lr * self.hat_m / (np.sqrt(self.hat_v) + self.eps)

            

