from value import Value
import random

class Neuron:

    def __init__(self, ni):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(ni)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        return sum((x*w for x, w in zip(x, self.w)), self.b).tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, ni, nn):
        self.neurons = [Neuron(ni) for _ in range(nn)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

    def __init__(self, ni, nns):
        inputs = [ni] + nns[:-1]
        self.layers = [Layer(ii, oi) for ii, oi in zip(inputs, nns)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
