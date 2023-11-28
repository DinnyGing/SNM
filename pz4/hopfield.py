import idx2numpy
import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = []
        for _ in range(10):
            self.weights.append(np.zeros((size, size)))

    def sign_function(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self, patterns, labels):
        for pattern, label in zip(patterns, labels):
            pattern = np.reshape(pattern, (self.size, 1))
            self.weights[label] += pattern @ pattern.T / len(patterns)
            np.fill_diagonal(self.weights[label], 0)  # діагональні елементи ваг встановлюються в 0

    def associate(self, input_pattern, weight, max_iterations=28):
        pattern = np.copy(input_pattern)
        for _ in range(max_iterations):
            rand_index = np.random.randint(0,self.size)
            index_activation = input_pattern * weight[rand_index,:]
            output_pattern = self.sign_function(index_activation)
            if np.array_equal(output_pattern, input_pattern):
                break
            pattern = output_pattern
        return pattern

    def association(self, input_pattern):
        outputs = []
        for i in range(10):
            outputs.append(self.associate(input_pattern, self.weights[i]))
        pattern = outputs[1]
        accuracy = 0
        for output in outputs:
            out_accuracy = self.accuracy(input_pattern, output)
            if accuracy < out_accuracy:
                accuracy = out_accuracy
                pattern = output
        return pattern


    def accuracy(self, input, output):
        correct = 0
        for x, y in zip(input, output):
            if x == y:
                correct += 1
        return correct / len(output)





# Приклад використання
size = 784
hopfield_net = HopfieldNetwork(size)

x_train = np.array(idx2numpy.convert_from_file("train-images.idx3-ubyte"))[0x10:].reshape((-1, 784))
y_train = np.array(idx2numpy.convert_from_file("train-labels.idx1-ubyte"))
x_test = np.array(idx2numpy.convert_from_file("t10k-images.idx3-ubyte"))[0x10:].reshape((-1, 784))
y_test = np.array(idx2numpy.convert_from_file("t10k-labels.idx1-ubyte"))

X_tain_binary = np.where(x_train>0, 1,-1)

x_train = []
labels_train = []
random_values = []
for i in range(10000):
    random_value = np.random.randint(len(X_tain_binary))
    while random_value in random_values:
        random_value = np.random.randint(len(X_tain_binary))
    random_values.append(random_value)
    x_train.append(X_tain_binary[random_value])
    labels_train.append(y_train[random_value])

hopfield_net.train(x_train, labels_train)

X_test_binary = np.where(x_test>0, 1,-1)

random_values = []
accuracies = []
for i in range(3000):
    random_value = np.random.randint(len(X_test_binary))
    while random_value in random_values:
        random_value = np.random.randint(len(X_test_binary))
    random_values.append(random_value)
    input_pattern = X_test_binary[random_value]
    output_pattern = hopfield_net.association(input_pattern)
    accuracies.append(hopfield_net.accuracy(input_pattern, output_pattern))


print("Accuracy: " + str(np.mean(accuracies)))