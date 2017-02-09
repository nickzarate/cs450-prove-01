from neuralNetwork.Neuron import Neuron


class NeuralNetwork(object):

    def __init__(self):
        # self.neurons = []
        # for i in range(num_neurons):
        #     self.neurons.append(Neuron())
        pass

    def train(self, data, targets):

        # Go through all the target values and create an array of all possible values
        # (Determines how many neurons we need)
        possible_targets = []
        for i in range(len(targets)):
            if targets[i] not in possible_targets:
                possible_targets.append(targets[i])

        # Create Neurons for each possible target value
        self.neurons = []
        for i in range(len(possible_targets)):
            self.neurons.append(Neuron(possible_targets[i], len(data[0]) + 1))

        # Go through each training instance and adjust the weights of each neuron accordingly
        for i in range(len(data)):
            for j in range(len(self.neurons)):
                self.neurons[j].setup(data[i])
                self.neurons[j].calculate_output()
                self.neurons[j].update_weights()

    def predict(self, data):
        pass
