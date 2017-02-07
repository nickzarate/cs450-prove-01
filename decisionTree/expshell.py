import pandas as pd
from sklearn import datasets
from decisionTree.DecisionTree import DecisionTree
import random


# Pre-process Iris Data
irises_raw_data = datasets.load_iris()
sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
for item in irises_raw_data.data:
    sepal_length.append(item[0])
    sepal_width.append(item[1])
    petal_length.append(item[2])
    petal_width.append(item[3])
irises_data = {
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width,
    'target': irises_raw_data.target
}
formatted_iris = pd.DataFrame(irises_data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
bins = 3
group_names = ['Short', 'Medium', 'Long']
formatted_iris['sepal_length'] = pd.cut(formatted_iris['sepal_length'], bins, labels=group_names)
formatted_iris['sepal_width'] = pd.cut(formatted_iris['sepal_width'], bins, labels=group_names)
formatted_iris['petal_length'] = pd.cut(formatted_iris['petal_length'], bins, labels=group_names)
formatted_iris['petal_width'] = pd.cut(formatted_iris['petal_width'], bins, labels=group_names)

# print("before importing lenses data")
# Pre-process lenses data
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data"
# lenses_data = pd.read_csv(url)#, columns = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'])
# print(lenses_data)
# print(type(lenses_data))
# formatted_lenses = pd.DataFrame(lenses_data, columns = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'])
# print(formatted_lenses)
# formatted_lenses = pd.DataFrame(lenses_data, columns = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'])
# print(formatted_lenses)

def randomize_data(data):
    new_data = []
    random.shuffle(data)
    split = int(len(data) * .7)
    training_set = data[:split:]
    testing_set = data[split::]
    testing_attributes = []
    testing_targets = []
    for i in range(len(testing_set)):
        testing_attribute = list(testing_set[i])
        testing_target = testing_attribute.pop()
        testing_attributes.append(testing_attribute)
        testing_targets.append(testing_target)

    new_data.append(training_set)
    new_data.append(testing_attributes)
    new_data.append(testing_targets)
    return new_data


def main(attribute_names, data):
    tree = DecisionTree(attribute_names)
    new_data = randomize_data(data)
    tree.train(new_data[0])

    # Calculate and print out the percent of predictions guessed correctly
    predictions = tree.predict(new_data[1])
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == new_data[2][i]:
            num_correct += 1
    percent_correct = num_correct / len(predictions)
    print("percent correct:", percent_correct)

main(["sepal_length", "sepal_width", "petal_length", "petal_width"], formatted_iris.values.tolist())
