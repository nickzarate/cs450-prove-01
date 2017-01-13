from sklearn import datasets
import random
irises_data = datasets.load_iris()


# Implements n-fold cross validation


class Iris(object):
  def __init__(self, sepal_length, sepal_width, petal_length, petal_width, target):
    self.sepal_length = sepal_length
    self.sepal_width = sepal_width
    self.petal_length = petal_length
    self.petal_width = petal_width
    self.target = target
  def test(self):
    print self.sepal_length
    print self.sepal_width
    print self.petal_length
    print self.petal_width
    print self.target


class HardCoded(object):
  def __init__(self):
    pass

  def train(self, data):
    pass

  def predict(self, data):
    predictions = []
    for item in data:
      predictions.append("setosa")
    return predictions


def find_percentage(training_sets, testing_set):
  hardCoded = HardCoded()
  for training_set in training_sets:
    hardCoded.train(training_set)

  # Create an array of data values to test and an array of target values to test
  testing_data_set = []
  testing_target_set = []
  for i in range(len(testing_set)):
    testing_data_set.append([testing_set[i].sepal_length, testing_set[i].sepal_width, testing_set[i].petal_length, testing_set[i].petal_width])
    testing_target_set.append(testing_set[i].target)

  right = 0
  predictions = hardCoded.predict(testing_data_set)
  for i in range(len(predictions)):
    if predictions[i] == testing_target_set[i]:
      right += 1

  percentage = right / float(len(testing_set)) * 100

  return percentage


def main():
  irises = []

  for i in range(len(irises_data.data)):
    iris = Iris(irises_data.data[i][0], irises_data.data[i][1], irises_data.data[i][2], irises_data.data[i][3], irises_data.target_names[irises_data.target[i]])
    irises.append(iris)

  random.shuffle(irises)

  original_sets = [irises[:30:], irises[30:60:], irises[60:90:], irises[90:120:], irises[120::]]
  percentages = []

  for i in range(5):
    sets = list(original_sets)
    testing_set = sets.pop(i)
    percentage = find_percentage(sets, testing_set)
    percentages.append(percentage)

  print sum(percentages) / len(percentages)

main()
