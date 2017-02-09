from neuralNetwork.NeuralNetwork import NeuralNetwork
from sklearn import datasets

irises_data = datasets.load_iris().data.tolist()
irises_targets = datasets.load_iris().target.tolist()


def main():
    network = NeuralNetwork()
    # print(irises_data.data)
    # print(irises_data.target)
    network.train(irises_data, irises_targets)
    pass

main()
