import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_to_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_to_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_to_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_to_output = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.hidden_raw = np.dot(input_data, self.weights_input_to_hidden) + self.bias_input_to_hidden
        # на 14 строчке переменная в которой будет применятся функция активации для self.hidden_raw для нормализации
        # данных
        self.hidden = self.tanh(self.hidden_raw)
        self.output_raw = np.dot(self.weights_hidden_to_output, self.hidden) + self.bias_hidden_to_output
        # на 19 строчке переменная в которой будет применятся функция активации для self.output_raw для нормализации
        # данных
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

# на 47 строці сігмоїдна функція активацї та на 50 - її похідна
    def sigmoid(S):
        return 1 / (1 + np.exp(-S))

    def sigmoid_backward(dA, S):
        sig = sigmoid(S)
        return dA * sig * (1 - sig)

# на 55 строці функція активацї ReLU та на 58 - її похідна
    def relu(RL):
        return np.maximum(0, RL)

    def relu_backward(dA, RL):
        dRL = np.array(dA, copy=True)
        dRL[RL <= 0] = 0
        return dRL

# на 64 строці функція активацї tanh та на 67 - її похідна
    def tanh(T):
        return np.tanh(T)

    def tanh_backward(dA, T):
        dT = tanh(T)
        return dA * (1 - np.square(dT))