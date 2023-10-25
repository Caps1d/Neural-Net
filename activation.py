import numpy as np


# Used in hidden layers
class ReLU:
    def forward(self, X: int):
        vec_max = np.vectorize(max)
        self.output = vec_max(0, X)
        return self.output

    def backward(self, X: int):
        pass


# Used in the output layer
class Softmax:
    def forward(self, X: int):
        exp_vals = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward(self, X: int):
        pass
