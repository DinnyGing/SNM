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

# Власне, поширення вперед. На 73 строці  - для огдного шару, а на 87 для всієї ШНМ.

    def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
        S_curr = np.dot(W_curr, A_prev) + b_curr

        if activation is "relu":
            activation_func = relu
        elif activation is "sigmoid":
            activation_func = sigmoid
        elif activation is "tanh":
            activation_func = tanh
        else:
            raise Exception('Non-supported activation function')

        return activation_func(S_curr), S_curr

    def full_forward_propagation(X, params_values, nn_architecture):
        memory = {}
        A_curr = X

        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_idx)]
            b_curr = params_values["b" + str(layer_idx)]
            A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory

# Це вже функції зворотнього поширення для градієнту. Розділені так само,
    # як і прямого поширення на 106 та 125 строках відповідно
    def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]

        if activation is "relu":
            backward_activation_func = relu_backward
        elif activation is "sigmoid":
            backward_activation_func = sigmoid_backward
        elif activation is "tanh":
            backward_activation_fun = tanh_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));

        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values


# оновлення значень параметрів, з двома масива для поточних значень параметрів
# та похідні функції витрат відповідно
def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

# функція втрат. Ця, можливо, не підійде, тоді просто видали/закоментуй. Бінарну кроссентропія
def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()