import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_to_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_to_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_to_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_to_output = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.hidden_raw = np.dot(input_data, self.weights_input_to_hidden) + self.bias_input_to_hidden
        # Примінюємо функцію активації для self.hidden_raw для нормалізації даних
        self.hidden = self.tanh(self.hidden_raw)
        # Транспонуємо self.hidden перед множенням
        self.output_raw = np.dot(self.hidden, self.weights_hidden_to_output) + self.bias_hidden_to_output
        # Примінюємо функцію активації для self.output_raw для нормалізації даних
        self.output = self.tanh(self.output_raw)
        return self.output

    def back_propagation(self, input_data, tags, learning_rate):
        # тут можно менять функции активации если это надо
        self.error = tags - self.output
        d_output = self.error
        # на 27 строке мы умножаем d_output.dot(self.weights_hidden_to_output.T) на производную функции
        # активации для скрытого слоя self.hidden вмест None
        d_hidden = d_output.dot(self.weights_hidden_to_output.T) * self.tanh_backward(self.hidden_raw)

        self.weights_hidden_to_output += self.hidden.T.dot(d_output) * learning_rate
        self.bias_hidden_to_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_to_hidden += input_data.T.dot(d_hidden) * learning_rate
        self.bias_input_to_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, input_data, tags, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward_propagation(input_data)
            self.back_propagation(input_data, tags, learning_rate)

    def predict(self, input_data):
        return self.forward_propagation(input_data)

# якщо будуть якісь помилки з функціями активації, хоч дзвоніть в будь який момент з 9 ранку до 10 ночі,
# раніше ао пізніше - пишіть

# Сігмоїдна функція активації
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

# Похідна сігмоїдної функції
    def sigmoid_backward(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

# ReLU функція активації
    def relu(self, RL):
        return np.maximum(0, RL)

# Похідна ReLU функції
    def relu_backward(self, Z):
        dZ = np.array(Z, copy=True)
        dZ[Z <= 0] = 0
        dZ[dZ > 1] = 1
        return dZ

# Гіперболічний тангенс функція активації
    def tanh(self, T):
        return np.tanh(T)

# Похідна гіперболічного тангенсу
    def tanh_backward(self, T):
        dT = self.tanh(T)
        return 1 - np.square(dT)

# Завантаження та підготовка даних
iris = load_iris()
X = iris.data
y = iris.target
y_one_hot = np.zeros((y.shape[0], 3))
y_one_hot[np.arange(y.shape[0]), y] = 1  # Перетворення у one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Параметри нейронної мережі
input_size = X_train.shape[1]
hidden_size = 4
output_size = y_one_hot.shape[1]
epochs = 1000
learning_rate = 0.1

# Створення та навчання нейронної мережі
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs, learning_rate)

# Передбачення на тестовому наборі
predictions = nn.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Визначення точності
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)
