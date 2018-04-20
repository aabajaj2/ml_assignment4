from knn import *
from sklearn.model_selection import train_test_split

df = load_data_set('data/PhishingData.arff')
trainSet, testSet = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
trainSet = trainSet.values.tolist()
testSet = testSet.values.tolist()

for k in range(2, 33):
    knn_clf = knn(k)
    knn_clf.fit(trainSet, trainSet)
    hyp = knn_clf.predict(testSet)
    score = get_accuracy(testSet, hyp)
    print("k=", k, 'Score=', score)