import time

import numpy as np
import pandas as pd

def normalize_data(data, attributes):
    for attribute in attributes:
        data[attribute] = round((data[attribute] - data[attribute].min())/
                                (data[attribute].max() - data[attribute].min()) * 1000)/1000
    return data

def mix_and_separate_data(data):
    n = data.shape[0]
    mixed_data = data.sample(n)
    n_train = round(n * 0.7)
    return {'train': mixed_data.iloc[:n_train], 'test': mixed_data.iloc[n_train:n]}


def assign_class_number(data, column):
    unique_values = data[column].unique()
    char_to_number = {}
    for i, char in enumerate(unique_values):
        char_to_number[char] = i
    data[column] = data[column].apply(lambda x: char_to_number[x])
    return data


def replace_na_value(data, column):
    average_value = data[column].mean(skipna=True)
    data[column].fillna(value=average_value, inplace=True)
    return data


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_backward(s):
    sig = sigmoid(s)
    return sig * (1 - sig)


def relu(x):
    x[x <= 0] = 0
    return x


def relu_backward(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def tanh(t):
    return np.tanh(t)


def tanh_backward(t):
    d_t = tanh(t)
    return 1 - np.square(d_t)


def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y = np.maximum(epsilon, np.minimum(1 - epsilon, y_true))  # Обмежимо значення y між epsilon та 1-epsilon
    return -np.mean(np.sum(y * np.log(y_pred), axis=1))


def predict(inputs):
    inputs_1 = np.dot(weights_0_1, inputs)
    for i in range(len(inputs_1)):
        inputs_1[i] += bias_0_1[i]
    outputs_1 = fun_l1(inputs_1)

    inputs_2 = np.dot(weights_1_2, outputs_1)
    for i in range(len(inputs_2)):
        inputs_2[i] += bias_1_2[i]
    outputs_2 = fun_l2(inputs_2)

    return outputs_2

def forward_propagation(inputs):
    inputs_1 = np.dot(weights_0_1, inputs) + bias_0_1
    outputs_1 = fun_l1(inputs_1)

    inputs_2 = np.dot(weights_1_2, outputs_1) + bias_1_2
    outputs_2 = fun_l2(inputs_2)
    return outputs_1, outputs_2


def back_propagation(outputs_1, outputs_2, inputs, expected_predict):
    error_layer_2 = np.array([outputs_2 - expected_predict])
    gradient_layer_2 = fun_back_l2(outputs_2)
    weights_delta_layer_2 = error_layer_2 * gradient_layer_2
    calc_weights_1_2 = weights_1_2 - learning_rate * (outputs_1.reshape(len(outputs_1), 1) @ weights_delta_layer_2).T
    calc_bias_1_2 = bias_1_2 - learning_rate * weights_delta_layer_2

    error_layer_1 = weights_delta_layer_2.dot(weights_1_2)
    gradient_layer_1 = fun_back_l1(outputs_1)
    weights_delta_layer_1 = error_layer_1 * gradient_layer_1
    calc_weights_0_1 = weights_0_1 - learning_rate * (inputs.reshape(len(inputs), 1) @ weights_delta_layer_1).T
    calc_bias_0_1 = bias_0_1 - learning_rate * weights_delta_layer_1

    return calc_weights_0_1, calc_weights_1_2, \
        calc_bias_0_1[0], calc_bias_1_2[0]


def sgd(X, y):
    calc_weights_1_2 = weights_1_2
    calc_weights_0_1 = weights_0_1
    calc_bias_1_2 = bias_1_2
    calc_bias_0_1 = bias_0_1
    for i in range(0, len(X), batch_size):
        # Forward pass
        batch_input = X[i:i+batch_size]
        batch_labels = y[i:i+batch_size]

        inputs_1 = np.dot(batch_input, weights_0_1.T)
        for j in range(len(inputs_1)):
            inputs_1[j] += bias_0_1
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, weights_1_2.T)
        for j in range(len(inputs_2)):
            inputs_2[j] += bias_1_2
        outputs_2 = fun_l2(inputs_2)

        # Calculate the error
        batch_labels_expected = []
        for out in batch_labels:
            batch_labels_expected.append(expect(out))
        batch_labels_expected = np.array(batch_labels_expected)
        error_layer_2 = outputs_2 - batch_labels_expected
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        calc_weights_1_2 -= learning_rate * (outputs_1.T @ weights_delta_layer_2).T
        for j in range(len(weights_delta_layer_2)):
            calc_bias_1_2 -= learning_rate * weights_delta_layer_2[j]

        error_layer_1 = weights_delta_layer_2.dot(weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)
        calc_weights_0_1 -= learning_rate * (batch_input.T @ weights_delta_layer_1).T
        for j in range(len(weights_delta_layer_1)):
            calc_bias_0_1 -= learning_rate * weights_delta_layer_1[j]
    return calc_weights_0_1, calc_weights_1_2, \
        calc_bias_0_1, calc_bias_1_2


def rmsprop_optimizer(X, y):
    calc_weights_1_2 = np.copy(weights_1_2)
    calc_weights_0_1 = np.copy(weights_0_1)
    calc_bias_1_2 = np.copy(bias_1_2)
    calc_bias_0_1 = np.copy(bias_0_1)

    epsilon = 1e-8
    rho = 0.9  # Decay factor

    moving_average_squared_1_2 = np.zeros_like(weights_1_2)
    moving_average_squared_0_1 = np.zeros_like(weights_0_1)

    for i in range(0, len(X), batch_size):

        batch_input = X[i:i+batch_size].reshape(-1, input_layer_count)
        batch_labels = y[i:i+batch_size]

        inputs_1 = np.dot(batch_input, calc_weights_0_1.T)
        for j in range(len(inputs_1)):
            inputs_1[j] += calc_bias_0_1
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, calc_weights_1_2.T)
        for j in range(len(inputs_2)):
            inputs_2[j] += calc_bias_1_2
        outputs_2 = fun_l2(inputs_2)

        batch_labels_expected = [expect(label) for label in batch_labels]
        error_layer_2 = outputs_2 - np.array(batch_labels_expected)
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        w2_sq = (outputs_1.T @ weights_delta_layer_2) ** 2
        moving_average_squared_1_2 = rho * moving_average_squared_1_2 + (1 - rho) * (w2_sq).T
        calc_weights_1_2 -= learning_rate * ((outputs_1.T @ weights_delta_layer_2).T / (np.sqrt(moving_average_squared_1_2) + epsilon))
        for j in range(len(weights_delta_layer_2)):
            calc_bias_1_2 -= learning_rate * weights_delta_layer_2[j]

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)

        w1_sq = (batch_input.T @ weights_delta_layer_1) ** 2
        moving_average_squared_0_1 = rho * moving_average_squared_0_1 + (1 - rho) * (w1_sq).T
        calc_weights_0_1 -= learning_rate * ((batch_input.T @ weights_delta_layer_1).T / (np.sqrt(moving_average_squared_0_1) + epsilon))
        for j in range(len(weights_delta_layer_1)):
            calc_bias_0_1 -= learning_rate * weights_delta_layer_1[j]

    return calc_weights_0_1, calc_weights_1_2, calc_bias_0_1, calc_bias_1_2


def adam_optimizer(X, y):
    calc_weights_1_2 = np.copy(weights_1_2)
    calc_weights_0_1 = np.copy(weights_0_1)
    calc_bias_1_2 = np.copy(bias_1_2)
    calc_bias_0_1 = np.copy(bias_0_1)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 0

    m_t_1_2 = np.zeros_like(weights_1_2)
    v_t_1_2 = np.zeros_like(weights_1_2)
    m_t_0_1 = np.zeros_like(weights_0_1)
    v_t_0_1 = np.zeros_like(weights_0_1)

    for i in range(0, len(X), batch_size):
        t += 1
        batch_input = X[i:i+batch_size]
        batch_labels = y[i:i+batch_size]

        inputs_1 = np.dot(batch_input, calc_weights_0_1.T)
        for j in range(len(inputs_1.T)):
            inputs_1.T[j] += calc_bias_0_1[j]
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, calc_weights_1_2.T)
        for j in range(len(inputs_2.T)):
            inputs_2.T[j] += calc_bias_1_2[j]
        outputs_2 = fun_l2(inputs_2)

        batch_labels_expected = [expect(label) for label in batch_labels]
        error_layer_2 = outputs_2 - np.array(batch_labels_expected)
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        m_t_1_2 = beta1 * m_t_1_2 + (1 - beta1) * (outputs_1.T @ weights_delta_layer_2).T
        v_t_1_2 = beta2 * v_t_1_2 + (1 - beta2) * ((outputs_1.T @ weights_delta_layer_2) ** 2).T

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)

        m_t_0_1 = beta1 * m_t_0_1 + (1 - beta1) * (batch_input.T @ weights_delta_layer_1).T
        v_t_0_1 = beta2 * v_t_0_1 + (1 - beta2) * ((batch_input.T @ weights_delta_layer_1) ** 2).T

        m_t_hat_1_2 = m_t_1_2 / (1 - beta1 ** t)
        v_t_hat_1_2 = v_t_1_2 / (1 - beta2 ** t)

        m_t_hat_0_1 = m_t_0_1 / (1 - beta1 ** t)
        v_t_hat_0_1 = v_t_0_1 / (1 - beta2 ** t)

        calc_weights_1_2 -= learning_rate * (m_t_hat_1_2 / (np.sqrt(v_t_hat_1_2) + epsilon))
        for j in range(len(weights_delta_layer_2)):
            calc_bias_1_2 -= learning_rate * weights_delta_layer_2[j]

        calc_weights_0_1 -= learning_rate * (m_t_hat_0_1 / (np.sqrt(v_t_hat_0_1) + epsilon))
        for j in range(len(weights_delta_layer_1)):
            calc_bias_0_1 -= learning_rate * weights_delta_layer_1[j]

        return calc_weights_0_1, calc_weights_1_2, calc_bias_0_1, calc_bias_1_2


def adagrad_optimizer(X, y):
    calc_weights_1_2 = np.copy(weights_1_2)
    calc_weights_0_1 = np.copy(weights_0_1)
    calc_bias_1_2 = np.copy(bias_1_2)
    calc_bias_0_1 = np.copy(bias_0_1)

    epsilon = 1e-8

    historical_gradient_squared_1_2 = np.zeros_like(weights_1_2)
    historical_gradient_squared_0_1 = np.zeros_like(weights_0_1)

    for i in range(0, len(X), batch_size):
        batch_input = X[i:i+batch_size]
        batch_labels = y[i:i+batch_size]

        # Прямое распространение
        inputs_1 = np.dot(batch_input, calc_weights_0_1.T)
        for j in range(len(inputs_1.T)):
            inputs_1.T[j] += calc_bias_0_1[j]
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, calc_weights_1_2.T)
        for j in range(len(inputs_2.T)):
            inputs_2.T[j] += calc_bias_1_2[j]
        outputs_2 = fun_l2(inputs_2)

        # Вычисление ошибки
        batch_labels_expected = [expect(label) for label in batch_labels]
        error_layer_2 = outputs_2 - np.array(batch_labels_expected)

        # Обратное распространение
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)

        # Обновление весов и смещений
        gradient_loss_1_2 = (outputs_1.T @ weights_delta_layer_2).T
        historical_gradient_squared_1_2 += (gradient_loss_1_2 ** 2)
        calc_weights_1_2 -= (learning_rate / (np.sqrt(historical_gradient_squared_1_2 + epsilon))) * gradient_loss_1_2
        for j in range(len(weights_delta_layer_2)):
            calc_bias_1_2 -= learning_rate * weights_delta_layer_2[j]

        gradient_loss_0_1 = (batch_input.T @ weights_delta_layer_1).T
        historical_gradient_squared_0_1 += (gradient_loss_0_1 ** 2)
        calc_weights_0_1 -= (learning_rate / (np.sqrt(historical_gradient_squared_0_1 + epsilon))) * gradient_loss_0_1
        for j in range(len(weights_delta_layer_1)):
            calc_bias_0_1 -= learning_rate * weights_delta_layer_1[j]

    return calc_weights_0_1, calc_weights_1_2, calc_bias_0_1, calc_bias_1_2


def train(inputs, expected_predict):
    outputs_1, outputs_2 = forward_propagation(inputs)
    return back_propagation(outputs_1, outputs_2, inputs, expected_predict)


def test(inputs, outputs):
    accuracy = 0

    for index, input in inputs.iterrows():
        labels = input
        actual = predict(np.array(labels))
        pos = outputs[index]
        for i in range(len(actual)):
            if actual[i] >= actual[pos]:
                pos = i
        if pos == outputs[index]:
            accuracy += 1
    return accuracy / len(outputs)


def expect(index):
    expected = np.zeros(output_layer_count)
    expected[index] = 1
    return expected


def get_attributes(columns, target):
    attributes = []
    for column in columns:
        if column != target:
            attributes.append(column)
    return attributes

wine_data = pd.read_csv('wine.data', header=None)
wine_data = assign_class_number(wine_data, 0)
wine_data = normalize_data(wine_data, get_attributes(wine_data.columns, 0))
mix_iris_data = mix_and_separate_data(wine_data)
train_data = mix_iris_data["train"]
train_labels = train_data[0]
train_data = train_data.drop(0, axis=1)
test_data = mix_iris_data["test"]
test_labels = test_data[0]
test_data = test_data.drop(0, axis=1)

hide_layer_count = 8
input_layer_count = 13
output_layer_count = 3
weights_0_1 = np.random.normal(-0.5, 0.5, (hide_layer_count, input_layer_count))
weights_1_2 = np.random.normal(-0.5, 0.5 ** -0.5, (output_layer_count, hide_layer_count))

bias_0_1 = np.zeros(hide_layer_count)
bias_1_2 = np.zeros(output_layer_count)

epochs = 50
learning_rate = 0.02
batch_size = 16
fun_l1 = tanh
fun_l2 = sigmoid
fun_back_l1 = tanh_backward
fun_back_l2 = sigmoid_backward

start_time = time.time()
for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for (_, labels), output in zip(train_data.iterrows(), train_labels):
        expected = expect(output)
        inputs_.append(np.array(labels))
        correct_predictions.append(np.array(expected))

    # here test
    weights_0_1, weights_1_2, \
        bias_0_1, bias_1_2 = adagrad_optimizer(np.array(train_data), train_labels)
    #here end test
    train_loss = categorical_crossentropy(np.array(correct_predictions),
                                          predict(np.array(inputs_).T).T)

    progress = 100 * e/float(epochs)
    if progress % 10 == 0:
        progress = str(progress)[:4]
        print("\rProgress: " + progress + " Training loss: ")
        print(train_loss)
end_time = time.time()

execution_time = end_time - start_time
print(f"Час виконання: {execution_time} секунд")

print("Accuracy: " + str(test(test_data, test_labels)))
