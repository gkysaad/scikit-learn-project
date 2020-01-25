from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class myKNN:
    def fit(self, x_train, y_train):
        self.train = x_train
        self.target = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best = euc(row, self.train[0])
        choice = self.target[0]
        for i in range(len(self.train)):
            new_dist = euc(row, self.train[i])
            if new_dist < best:
                best = new_dist
                choice = self.target[i]
        return choice



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

my_classifier = myKNN()

my_classifier.fit(x_train, y_train)

pred = my_classifier.predict(x_test)
print(accuracy_score(y_test, pred))