from neuralNetwork.Neuron import Neuron
import numpy


class NeuralNetwork(object):

    def __init__(self, learning_rate, num_neurons, num_attributes, possible_targets):
        self.neurons = []
        self.learning_rate = learning_rate
        self.num_neurons = []
        self.possible_targets = possible_targets
        num_weights = num_attributes
        for i in range(len(num_neurons)):
            layer = []
            for j in range(num_neurons[i]):
                layer.append(Neuron(num_weights))
            self.neurons.append(layer)
            num_weights = num_neurons[i] + 1

    def train(self, data, targets):
        # Go through each training instance and adjust the weights of each neuron accordingly
        # For each instance in my data set...
        for i in range(len(data)):
            input_values = data[i]

            # Feed forward ------------------------------------------------------------------
            # For each layer in my neural network...
            for j in range(len(self.neurons)):
                temp_input_values = []
                # For each neuron in the current layer...
                for k in range(len(self.neurons[j])):
                    # For each neuron, find the activation
                    self.neurons[j][k].calculate_output(input_values)
                    temp_input_values.append(self.neurons[j][k].get_activation())
                input_values = list(temp_input_values)

            # Back Propagation --------------------------------------------------------------
            # For each layer in my network (reversed)
            for j, layer in reversed(list(enumerate(self.neurons))):
                # Go through each node and update the "new_weights" for each one...
                for k in range(len(self.neurons[j])):
                    activation = self.neurons[j][k].get_activation()
                    error = activation * (1 - activation)
                    # If we are on the output layer...
                    if j == (len(self.neurons) - 1):
                        # Calculate error with activation and target value
                        target = 1 if self.possible_targets[k] == targets[i] else 0
                        error *= (activation - target)
                        self.neurons[j][k].set_error(error)
                    else:
                        error_weights = 0
                        # Iterate through layer on the right
                        for l in range(len(self.neurons[j + 1])):
                            temp_error = self.neurons[j + 1][l].get_error() * self.neurons[j + 1][l].get_weight(k)
                            error_weights += temp_error
                        error *= error_weights
                        self.neurons[j][k].set_error(error)
                    self.neurons[j][k].set_new_weights(self.learning_rate)

            # Update all the weights for good before moving onto the next training instance
            for j in range(len(self.neurons)):
                for k in range(len(self.neurons[j])):
                    self.neurons[j][k].update_weights()

    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            input_values = data[i]
            for j in range(len(self.neurons)):
                output_values = []
                for k in range(len(self.neurons[j])):
                    self.neurons[j][k].calculate_output(input_values)
                    output_values.append(self.neurons[j][k].get_activation())
                input_values = list(output_values)
                if j == len(self.neurons) - 1:
                    prediction_index = numpy.argmax(input_values)
                    predictions.append(self.possible_targets[prediction_index])
                    break
        return predictions
