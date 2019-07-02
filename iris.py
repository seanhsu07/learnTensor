import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

testIdx = [0, 50, 100]  # indexes of one of each kind of flower

# training data
train_target = np.delete(iris.target, testIdx)
train_data = np.delete(iris.data, testIdx, axis=0)

# testing data
test_target = iris.target[testIdx]
test_data = iris.data[testIdx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))
