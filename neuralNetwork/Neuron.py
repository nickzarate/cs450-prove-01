import random
import math


class Neuron(object):

    def __init__(self, num_weights):
        self.activation = 0
        self.input_values = []
        self.weights = []
        self.target = 0
        self.error = 0
        for i in range(num_weights):
            self.weights.append(random.uniform(-1.0, 1.0))
        self.new_weights = list(self.weights)

    def get_activation(self):
        return self.activation

    def set_error(self, error):
        self.error = error

    def get_error(self):
        return self.error

    def get_weight(self, index):
        return self.weights[index]

    def calculate_output(self, input_values):
        self.activation = 0
        self.input_values = list(input_values)
        self.input_values.insert(0, -1.0)
        for i in range(len(self.input_values)):
            self.activation += self.input_values[i] * self.weights[i]
        self.activation = 1.0 / (1.0 + (math.pow(math.e, -self.activation)))

    def set_new_weights(self, learning_rate):
        for i in range(len(self.new_weights)):
            self.new_weights[i] -= (learning_rate * self.error * self.input_values[i])

    def update_weights(self):
        self.weights = list(self.new_weights)