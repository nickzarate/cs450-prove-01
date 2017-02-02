import pandas as pd
from sklearn import datasets
from decisionTree.DecisionTree import DecisionTree


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
print(formatted_iris)
print(pd.value_counts(formatted_iris['sepal_length']))


def main():
    tree = DecisionTree()
    tree.train(formatted_iris)

main()
