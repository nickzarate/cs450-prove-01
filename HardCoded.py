import numpy


class HardCoded(object):
    data = []
    targets = []
    outcomes = [0, 1, 2]

    def __init__(self):
        self.mean = []
        self.stddev = []

    def train(self, data, targets):
        self.data = data
        self.targets = targets
        self.find_mean_std()
        pass

    def find_mean_std(self):
        if len(self.data) > 0:
            # Find the mean and standard deviation of the data
            for i in range(len(self.data[0])):
                arr = []
                for object in self.data:
                    arr.append(object[i])
                self.mean.append(numpy.mean(arr))
                self.stddev.append(numpy.std(arr))
        self.normalize_data()

    def normalize_data(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.data[i][j] = (self.data[i][j] - self.mean[j]) / self.stddev[j]

    def predict(self, data, num_neighbors):
        predictions = []
        distances = []
        for i in range(len(data)):
            # Normalize the data being predicted before calculating nearest neighbors
            for j in range(len(data[i])):
                data[i][j] = (data[i][j] - self.mean[j]) / self.stddev[j]

            # Iterate through training data to find nearest neighbors.
            nearest_neighbors = [len(data) + 1] * num_neighbors
            distances = []
            for j in range(len(self.data)):
                distance = 0
                for k in range(len(self.data[j])):
                    distance += numpy.math.pow((data[i][k] - self.data[j][k]), 2)
                distances.append(distance)
                for l in range(len(nearest_neighbors)):
                    if nearest_neighbors[l] > len(data) or distance < distances[nearest_neighbors[l]]:
                        nearest_neighbors[l] = j
                        break

            prediction = [0] * 3
            for j in range(len(nearest_neighbors)):
                if self.targets[nearest_neighbors[j]] == self.outcomes[0]:
                    prediction[0] += 1
                elif self.targets[nearest_neighbors[j]] == self.outcomes[1]:
                    prediction[1] += 1
                else:
                    prediction[2] += 1

            highest = max(prediction)
            for j in range(len(prediction)):
                if prediction[j] == highest:
                    predictions.append(self.outcomes[j])
                    break

        return predictions

# normalize the data, then use the euclidean distance metric
