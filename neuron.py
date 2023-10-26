import numpy as np

from activation import ReLU, Softmax
from loss import CategoricalCrossEntropy

np.random.seed(0)


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, activation: object):
        # n_inputs x n_neurons = T(n_neurons x n_inputs)
        # Hence we take care of transpose for the dot product
        # 0.10 * to ensure we're between 0.1 and -.1
        self.activation = activation()
        self.size = n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: list):
        output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(output)
        return self.output


def main():
    X = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]
    layer1 = LayerDense(len(X[0]), 5, activation=ReLU)
    layer2 = LayerDense(layer1.size, 2, activation=ReLU)
    layer3 = LayerDense(layer2.size, 2, activation=Softmax)

    # activation = ReLU()

    layer1.forward(X)

    layer2.forward(layer1.output)

    layer3.forward(layer2.output)

    print(layer1.output)
    print(layer2.output)
    print(layer3.output)

    loss = CategoricalCrossEntropy()
    # probs = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
    # y_pred = [0.2, 0.5, 0.3]
    y_pred = layer3.output
    y_true = [0, 1, 0]
    print(loss.calculate(y_pred, y_true))
    pass


if __name__ == "__main__":
    main()
