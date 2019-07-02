from sklearn import datasets
from scipy.spatial import distance
import random

def euc(a,b):
    return distance.euclidean(a,b)
    pass

class scrappyKNN():
    """docstring for scrappyKNN."""
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
            pass
        return predictions
    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
                pass
            pass
        return self.y_train[best_index]


iris = datasets.load_iris()

x = iris.data
y = iris.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# second classifier
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

# first classifier
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# Own KNeighborsClassifier
my_classifier = scrappyKNN()

my_classifier.fit(x_train, y_train)
prediction = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))
