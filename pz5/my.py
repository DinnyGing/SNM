import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class LVQ:
    def __init__(self, n, c, alpha=0.017, epochs=20):
        self.alpha = alpha
        self.epochs = epochs
        self.weights = np.random.rand(c, n)

    def euclidean_distances(self, input):
        d = []
        for i in range(len(self.weights)):
            d.append(np.sum((input - self.weights[i])** 2))
        return d

    def best_match(self, input):
        return np.argmin(self.euclidean_distances(input))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                win = self.best_match(xi)
                if win == yi:
                    self.weights[win] += self.alpha * (xi - self.weights[win])
                else:
                    self.weights[win] -= self.alpha * (xi - self.weights[win])

    def predict(self, X):
        values = []
        for xi in X:
            win = self.best_match(xi)
            values.append(win)
        return values


def accuracy(y_test, y_pred):
    accuracy = 0
    for y_t, y_p in zip(y_test, y_pred):
        if y_t == y_p:
            accuracy += 1
    return accuracy / len(y_test)


def recall(y_test, y_pred, n):
    recall = []
    for i in range(n):
        correct = 0
        false = 0
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == i and y_t == y_p:
                correct += 1
            elif y_p == i and y_t != y_p:
                false += 1
        recall.append(correct/ (correct + false))

    return np.mean(recall)


def precision(y_test, y_pred, n):
    recall = []
    for i in range(n):
        correct = 0
        classes = 0
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == i:
                if y_t == y_p:
                    correct += 1
                classes += 1
        recall.append(correct/ classes)

    return np.mean(recall)



iris_data = load_iris()
data, target = shuffle(iris_data.data, iris_data.target)
scaler = StandardScaler()
data = scaler.fit_transform(data)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)


lvq = LVQ(4, 3)
lvq.train(x_train, y_train)
y_pred = lvq.predict(x_test)
accuracy = accuracy(y_test, y_pred)
precision = precision(y_test, y_pred, 3)
recall = recall(y_test, y_pred, 3)
f1 = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)