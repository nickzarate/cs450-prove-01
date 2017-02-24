import random
import pandas
import numpy


class Preprocessor(object):

    def __init__(self, data):
        # Randomize the data
        random.shuffle(data)
        self.full_data = data

        # Separate the data into "data" and "targets"
        self.targets = []
        self.data = []
        for i in range(len(data)):
            instance = list(data[i])
            target = instance.pop()
            self.targets.append(target)
            self.data.append(instance)

        # Normalize the data
        self.data = pandas.DataFrame(self.data)
        self.data = self.data.apply(lambda x: (x - numpy.mean(x)) / (numpy.max(x) - numpy.min(x)))
        self.data = self.data.values.tolist()

    def get_data(self, training_percent, validation_percent, testing_percent):
        # Check to see if the percentages all add up to 100
        if (training_percent + validation_percent + testing_percent) != 100:
            return []
        training_split = int(len(self.data) * (training_percent / 100.0))
        validation_split = int(training_split + len(self.data) * (validation_percent / 100.0))

        # Create data/target pairs for training, validation, and testing
        training_data = self.data[:training_split:]
        training_targets = self.targets[:training_split:]
        validation_data = self.data[training_split:validation_split:]
        validation_targets = self.targets[training_split:validation_split:]
        testing_data = self.data[validation_split::]
        testing_targets = self.targets[validation_split::]

        # Return values in array
        return [training_data, training_targets, validation_data, validation_targets, testing_data, testing_targets]
