from neuralNetwork.Neuron import Neuron


class NeuralNetwork(object):

    def __init__(self):
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
        print("before updating the weights of each neuron")
        for i in range(len(data)):
            for j in range(len(self.neurons)):
                self.neurons[j].set_target(targets[i])
                print("instance is a:", targets[i])
                self.neurons[j].setup(data[i])
                self.neurons[j].calculate_output()
                print("weights before update for neuron", j, self.neurons[j].weights)
                self.neurons[j].update_weights()
                print("weights after update for neuron", j, self.neurons[j].weights)

    def predict(self, data):
        predictions = []
        prediction = 0
        for instance in data:
            for neuron in self.neurons:
                neuron.setup(instance)
                neuron.calculate_output()
                if neuron.activation > 0.5:
                    prediction = neuron.neuron_name
                    break
            predictions.append(prediction)
        return predictions
