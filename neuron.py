import numpy as np


class Neuron:
    weights = []
    bias = 0

    def __init__(self, weights: int, bias):
        self.weights = weights
        self.bias = bias

    def output(self, input: int):
        return np.dot(self.weights, input) + self.bias


class Layer:
    size = 0
    neurons = []

    def __init__(self, n: int, weights, biases):
        self.size = n
        self.neurons = [Neuron(weights[i], biases[i]) for i in range(n)]

    def output(self, input: int):
        outputs = []
        for i in range(len(self.neurons)):
            outputs.append(self.neurons[i].output(input))

        return outputs
