from neuralNetwork.NeuralNetwork import NeuralNetwork
from sklearn import datasets
import random
import pandas
import numpy


# Pre-process the Iris Data
irises_data = datasets.load_iris().data.tolist()
irises_targets = datasets.load_iris().target.tolist()
# print(irises_data)
# print(irises_targets)

full_data = []
for i in range(len(irises_data)):
    full_instance = list(irises_data[i] + [irises_targets[i]])
    full_data.append(full_instance)
random.shuffle(full_data)
for i in range(len(full_data)):
    instance = list(full_data[i])
    instance_target = instance.pop()
    instance_data = list(instance)
    irises_data[i] = instance_data
    irises_targets[i] = instance_target

irises_data = pandas.DataFrame(irises_data)
print(irises_data)
irises_data = irises_data.apply(lambda x: (x - numpy.mean(x)) / (numpy.max(x) - numpy.min(x)))
print(irises_data)
irises_data = irises_data.values
irises_data = irises_data.tolist()


split = int(len(irises_data) * 0.7)
iris_training_data = irises_data[:split:]
iris_training_targets = irises_targets[:split:]
iris_testing_data = irises_data[split::]
iris_testing_targets = irises_targets[split::]


def main():
    network = NeuralNetwork()
    network.train(iris_training_data, iris_training_targets)
    predictions = network.predict(iris_testing_data)
    print(predictions)
    print(iris_testing_targets)

    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == iris_testing_targets[i]:
            num_correct += 1
    percent_correct = num_correct / len(predictions)
    print("percent correct:", percent_correct)

main()
