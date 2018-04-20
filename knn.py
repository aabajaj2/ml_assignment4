from classifier import classifier
import operator
import math
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split


def loadDataset(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    df = df.astype('int')
    return df

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(train_x, test_x, k):
    distances = []
    length = len(test_x) - 1
    for x in range(len(train_x)):
        dist = euclideanDistance(test_x, train_x[x], length)
        distances.append((train_x[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


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
            result = get_response(neighbors)
            hypothesis.append(result)
        return hypothesis


if __name__ == '__main__':
    df = loadDataset('data/PhishingData.arff')
    trainSet, testSet = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
    trainSet = trainSet.values.tolist()
    testSet = testSet.values.tolist()

    for k in range(2, 12):
        knnclf = knn(k)
        knnclf.fit(trainSet, trainSet)
        hyp = knnclf.predict(testSet)
        score = get_accuracy(testSet, hyp)
        print('Score=', score, "k=", k)