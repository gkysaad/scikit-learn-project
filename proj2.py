from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class myKNN:
    def fit(self, x_train, y_train):
        self.train = x_train
        self.target = y_train

    def predict(self, k, x_test):
        predictions = []
        for row in x_test:
            neighbours = self.closest(row)
            counter = {}
            if len(neighbours) < k:
                k = len(neighbours)
            for i in range(k):
                if str(neighbours[i][1]) in counter:
                    counter[str(neighbours[i][1])] += 1
                else:
                    counter[str(neighbours[i][1])] = 1
            #print(counter)
            counter = sorted(counter, key=counter.get)
            predictions.append(int(counter[0]))
        #print(predictions)                   
        return predictions
    
    def closest(self, row):
        arr = []
        for i in range(len(self.train)):
            new_dist = euc(row, self.train[i])
            arr.append((new_dist, self.target[i]))
        return sorted(arr)



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

pred = my_classifier.predict(3, x_test)
print(accuracy_score(y_test, pred))