import random


class Neuron(object):

    def __init__(self, target, num_weights):
        self.activation = 0
        self.target = target
        self.weights = []
        self.learning_rate = 0.15
        for i in range(num_weights):
            self.weights.append(random.uniform(-1.0, 1.0))
        pass

    def setup(self, instance):
        bias = list(instance)
        bias.insert(0, -1.0)
        self.input_values = bias
        print(self.weights)
        print(self.input_values)

    def calculate_output(self):
        for i in range(len(self.input_values)):
            self.activation += self.input_values[i] * self.weights[i]

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (self.learning_rate * (self.activation - self.target) * self.input_values[i])
