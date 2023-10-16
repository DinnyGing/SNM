import time

import numpy as np
import pandas as pd

def normalize_data(data, attributes):
    for attribute in attributes:
        min = data[attribute].min()
        max = data[attribute].max()
        data[attribute] = round((data[attribute] - min)/(max - min) * 1000)/1000
    return data


def mix_and_separate_data(data):
    n = data.shape[0]
    mixed_data = data.sample(n)
    n_train = round(n * 0.7)
    train_data = mixed_data.iloc[:n_train]
    test_data = mixed_data.iloc[n_train:n]
    return {'train': train_data, 'test': test_data}


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
    dT = tanh(t)
    return 1 - np.square(dT)


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


def train(inputs, expected_predict):
    inputs_1 = np.dot(weights_0_1, inputs) + bias_0_1
    outputs_1 = fun_l1(inputs_1)

    inputs_2 = np.dot(weights_1_2, outputs_1) + bias_1_2
    outputs_2 = fun_l2(inputs_2)

    actual_predict = outputs_2
    error_layer_2 = np.array([actual_predict - expected_predict])
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


def test(inputs, outputs):
    accuracy = 0
    for (_, labels), output in zip(inputs.iterrows(), outputs):
        actual = predict(np.array(labels))
        pos = output
        for i in range(len(actual)):
            if actual[i] >= actual[pos]:
                pos = i
        if pos == output:
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



iris_data = pd.read_csv('iris.data', header=None)
iris_data = iris_data.rename(columns={0: "sepal length", 1: "sepal width", 2: "petal length", 3: "petal width", 4: "class"})
iris_data = assign_class_number(iris_data, "class")
iris_data = normalize_data(iris_data, get_attributes(iris_data.columns, "class"))
mix_iris_data = mix_and_separate_data(iris_data)
train_iris_data = mix_iris_data["train"]
train_iris_output = train_iris_data["class"]
train_iris_data = train_iris_data.drop("class", axis=1)
test_iris_data = mix_iris_data["test"]
test_iris_output = test_iris_data["class"]
test_iris_data = test_iris_data.drop("class", axis=1)

hide_layer_count = 4
input_layer_count = len(train_iris_data.columns)
output_layer_count = len(train_iris_output.unique())
weights_0_1 = np.random.normal(-0.5, 0.5, (hide_layer_count, input_layer_count))
weights_1_2 = np.random.normal(-0.5, 0.5 ** -0.5, (output_layer_count, hide_layer_count))

bias_0_1 = np.zeros(hide_layer_count)
bias_1_2 = np.zeros(output_layer_count)

epochs = 100
learning_rate = 0.07
fun_l1 = tanh
fun_l2 = relu
fun_back_l1 = tanh_backward
fun_back_l2 = relu_backward

start_time = time.time()
for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for (_, labels), iris_output in zip(train_iris_data.iterrows(), train_iris_output):
        expected = expect(iris_output)
        weights_0_1, weights_1_2, \
            bias_0_1, bias_1_2 = train(np.array(labels), expected)
        inputs_.append(np.array(labels))
        correct_predictions.append(np.array(expected))

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

print("Accuracy: " + str(test(test_iris_data, test_iris_output)))

