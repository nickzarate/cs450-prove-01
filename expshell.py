from HardCoded import HardCoded
from Iris import Iris
import random
from functools import reduce
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

irises_data = datasets.load_iris()


def find_percentage(training_set, testing_set):
    hard_coded = HardCoded()

    training_data_set = []
    training_target_set = []
    for iris in training_set:
        training_data_set.append(iris.data)
        training_target_set.append(iris.target)

    # Create an array of data values to test and an array of target values to test
    testing_data_set = []
    testing_target_set = []
    for iris in testing_set:
        testing_data_set.append(iris.data)
        testing_target_set.append(iris.target)

    hard_coded.train(training_data_set, training_target_set)

    right = 0
    predictions = hard_coded.predict(testing_data_set, 4)
    for i in range(len(predictions)):
        if predictions[i] == testing_target_set[i]:
            right += 1

    percentage = right / float(len(testing_set)) * 100

    # print("my percentage:")
    # print(percentage)

    return percentage


def find_sk_percentage(training_set, testing_set):

    training_data_set = []
    training_target_set = []
    for iris in training_set:
        training_data_set.append(iris.data)
        training_target_set.append(iris.target)

    # Create an array of data values to test and an array of target values to test
    testing_data_set = []
    testing_target_set = []
    for iris in testing_set:
        testing_data_set.append(iris.data)
        testing_target_set.append(iris.target)

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(training_data_set, training_target_set)
    sk_right = 0
    sk_predictions = classifier.predict(testing_data_set)
    for i in range(len(sk_predictions)):
        if sk_predictions[i] == testing_target_set[i]:
            sk_right += 1

    sk_percentage = sk_right / float(len(testing_set)) * 100

    # print("sk's percentage:")
    # print(sk_percentage)

    return sk_percentage



def main():
    irises = []

    for i in range(len(irises_data.data)):
        iris = Iris(irises_data.data[i], irises_data.target[i])
        irises.append(iris)

    random.shuffle(irises)

    original_sets = [irises[:30:], irises[30:60:], irises[60:90:], irises[90:120:], irises[120::]]
    percentages = []
    sk_percentages = []

    for i in range(5):
        sets = list(original_sets)
        testing_set = sets.pop(i)
        training_set = reduce((lambda x, y: x + y), sets)

        # Find percent accurate using mine
        percentage = find_percentage(training_set, testing_set)
        percentages.append(percentage)

        # Find percent accurate using scikit-learn
        sk_percentage = find_sk_percentage(training_set, testing_set)
        sk_percentages.append(sk_percentage)

    print("My percentage:")
    print(sum(percentages) / len(percentages))
    print("Scikit learn's percentage:")
    print(sum(sk_percentages) / len(sk_percentages))

main()
