import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class LVQ:
    def __init__(self, alpha=0.01, epochs=20):
        self.alpha = alpha
        self.epochs = epochs
        self.w2 = np.array([[1, 0, 0], # for iris dataset
              [0, 1, 0],
              [0, 0, 1]])
    def train(self, X, y):
        y = pd.Series(y)
        X = np.array(X)
        self.w1 = np.random.rand(self.w2.shape[0], len(X[0, :]))
        for epoch in range(self.epochs):
            for i in range(len(X)):
                p_temp = np.array(X[i, :]).reshape(-1, 1)
                n1 = np.zeros((self.w1.shape[0]))
                for j in range(self.w1.shape[0]):
                    n1[j] = -np.linalg.norm(self.w1[j].reshape(-1, 1) - p_temp)
                a1 = compet(n1)
                a2 = self.w2 @ a1.reshape(-1, 1)
                winner_neuron = np.argmax(a1)
                out = np.argmax(a2)
                if out == y.iloc[i]:
                    self.w1[winner_neuron] += self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                else:
                    self.w1[winner_neuron] -= self.alpha * (p_temp.ravel() - self.w1[winner_neuron])
                    k = len(n1)-1
                    while True:
                        a1 = np.zeros(n1.shape)
                        ele = n1.argsort()[k]
                        a1[ele] = 1
                        a2 = self.w2 @ a1.reshape(-1, 1)
                        out = np.argmax(a2)
                        if out == y.iloc[i]:
                            self.w1[ele] += self.alpha * (p_temp.ravel() - self.w1[ele])
                            break
                        else:
                            k -= 1
        return self


    def test(self, X, y):
        accuracy = 0
        y = pd.Series(y)
        X = np.array(X)
        for i in range(len(X)):
            p_temp = np.array(X[i, :]).reshape(-1, 1)
            n1 = np.zeros((self.w1.shape[0]))
            for j in range(self.w1.shape[0]):
                n1[j] = -np.linalg.norm(self.w1[j].reshape(-1, 1) - p_temp)
            a1 = compet(n1)
            a2 = self.w2 @ a1.reshape(-1, 1)
            out = np.argmax(a2)
            if out == y.iloc[i]:
                accuracy += 1
        return accuracy / len(X)
    def predict(self, X):
        values = []
        X = np.array(X)
        for i in range(len(X)):
            p_temp = np.array(X[i, :]).reshape(-1, 1)
            n1 = np.zeros((self.w1.shape[0]))
            for j in range(self.w1.shape[0]):
                n1[j] = -np.linalg.norm(self.w1[j].reshape(-1, 1) - p_temp)
            a1 = compet(n1)
            a2 = self.w2 @ a1.reshape(-1, 1)
            out = np.argmax(a2)
            values.append(out)
        return values

def compet(x):
    max_val = np.max(x)
    res_arr = np.zeros_like(x)
    res_arr[x == max_val] = 1
    return res_arr


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

# Розділити дані на train і test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

lvq = LVQ()
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
