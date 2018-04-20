from classifier import classifier
import operator
import math
from scipy.io import arff
import pandas as pd


def load_data_set(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    df = df.astype('int')
    return df


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(train_x, test_x, k):
    distances = []
    length = len(test_x) - 1
    for x in range(len(train_x)):
        dist = euclidean_distance(test_x, train_x[x], length)
        distances.append((train_x[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_predictions(neighbors):
    neighs = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in neighs:
            neighs[response] += 1
        else:
            neighs[response] = 1
    sorted_neighs = sorted(neighs.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_neighs[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return correct/float(len(test_set))


class knn(classifier):

    def __init__(self, k):
        super().__init__()
        self.train_x = []
        self.k = k

    def fit(self, X, Y):
        self.train_x = X

    def predict(self, X):
        hypothesis = []
        for x in range(len(X)):
            neighbors = get_neighbors(self.train_x, X[x], self.k)
            result = get_predictions(neighbors)
            hypothesis.append(result)
        return hypothesis
