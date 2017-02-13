import random


class Neuron(object):

    def __init__(self, target, num_weights):
        self.activation = 0
        self.neuron_name = target
        self.weights = []
        self.learning_rate = 0.3
        self.target = 0
        for i in range(num_weights):
            self.weights.append(random.uniform(-1.0, 1.0))
        pass

    def set_target(self, target):
        if self.neuron_name == target:
            self.target = 1
        else:
            self.target = 0

    def setup(self, instance):
        bias = list(instance)
        bias.insert(0, -1.0)
        self.input_values = bias
        print("weights and input values")
        print(self.weights)
        print(self.input_values)

    def calculate_output(self):
        self.activation = 0
        for i in range(len(self.input_values)):
            self.activation += self.input_values[i] * self.weights[i]
        print("activation", self.activation)

    def update_weights(self):
        print(self.target)
        print(self.learning_rate)
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * (self.activation - self.target) * self.input_values[i])
