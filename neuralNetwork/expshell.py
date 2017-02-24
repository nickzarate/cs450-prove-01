from neuralNetwork.NeuralNetwork import NeuralNetwork
from sklearn import datasets
import random
import pandas
import numpy
from neuralNetwork.Preprocessor import Preprocessor
from sklearn.neural_network import MLPClassifier


# Pre-process Iris Data #2
# Create 2D array
irises_data = datasets.load_iris().data.tolist()
irises_targets = datasets.load_iris().target.tolist()
full_data = []
for i in range(len(irises_data)):
    full_instance = list(irises_data[i] + [irises_targets[i]])
    full_data.append(full_instance)
# Pre-process 2D array
preprocessor = Preprocessor(full_data)
full_iris = preprocessor.get_data(50, 25, 25)


# Pre-process Pima #2
pima = pandas.read_csv("./pima.csv")
pima = pima.sample(frac=1).reset_index(drop=True)
pima = pima.as_matrix().tolist()
pima_processor = Preprocessor(pima)
full_pima = pima_processor.get_data(50, 25, 25)


# Return an array that represents the "set" of items in the training
def get_possible_targets(training_targets):
    possible_targets = []
    for target in training_targets:
        if target not in possible_targets:
            possible_targets.append(target)
    return possible_targets


def compare_predictions(predictions, targets):
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            num_correct += 1
    percent_correct = num_correct / len(predictions)
    return percent_correct * 100


def find_percentage(network, data, targets):
    predictions = network.predict(data)
    return compare_predictions(predictions, targets)


# Create and train the neural network, and then test it and return the percentage it got correct
def get_percent_correct(learning_rate, epochs, num_neurons, training_data, training_targets,
                        validation_data, validation_targets, testing_data, testing_targets):
    possible_targets = get_possible_targets(training_targets)
    num_neurons.append(len(possible_targets))
    network = NeuralNetwork(learning_rate, num_neurons, len(training_data[0]) + 1, possible_targets)
    for i in range(epochs):
        network.train(training_data, training_targets)
        # print(find_percentage(network, validation_data, validation_targets))
    return find_percentage(network, testing_data, testing_targets)


# Set parameters for neural network and then call get_percent_correct for each data set
def main():
    iris_learning_rate = 0.25
    iris_num_neurons = [4]
    iris_epochs = 200
    percent_iris = get_percent_correct(iris_learning_rate, iris_epochs, iris_num_neurons, full_iris[0],
                                       full_iris[1], full_iris[2], full_iris[3], full_iris[4], full_iris[5])
    print(percent_iris)

    pima_learning_rate = 0.25
    pima_num_neurons = [12, 5]
    pima_epochs = 700
    percent_pima = get_percent_correct(pima_learning_rate, pima_epochs, pima_num_neurons, full_pima[0],
                                       full_pima[1], full_pima[2], full_pima[3], full_pima[4], full_pima[5])
    print(percent_pima)

main()
