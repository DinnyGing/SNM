import numpy as np
import pandas as pd


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


iris_data = pd.read_csv('iris.data', header=None)
iris_data = iris_data.rename(
    columns={0: "sepal length", 1: "sepal width", 2: "petal length", 3: "petal width", 4: "class"})
iris_data = assign_class_number(iris_data, "class")
mix_iris_data = mix_and_separate_data(iris_data)
train_iris_data = mix_iris_data["train"]
train_iris_output = train_iris_data["class"]
train_iris_data = train_iris_data.drop("class", axis=1)
test_iris_data = mix_iris_data["test"]
test_iris_output = test_iris_data["class"]
test_iris_data = test_iris_data.drop("class", axis=1)

weight_input_to_hidden = np.random.uniform(-0.5, 0.5, (4,))
weight_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 4))

bias_input_to_hidden = np.zeros((4, 4))
bias_hidden_to_output = np.zeros((10, 4))

epochs = 20
e_loss = 0
e_correct = 0
learning_rate = 1

for epochs in range(epochs):
    print(f"Epochs №{epochs}")

    for (_, labels), iris_output in zip(train_iris_data.iterrows(), train_iris_output):
        labels = np.array(labels)
        # Forward
        hidden_raw = weight_input_to_hidden @ labels + bias_input_to_hidden
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

        output_raw = weight_hidden_to_output @ hidden + bias_hidden_to_output
        output = 1 / (1 + np.exp(-output_raw))
        # print(output)
        # Loss / Error calculation
        e_loss += 1 / len(output) * np.sum((output - iris_output) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(iris_output))

        # Backpropogation (output layer)
        delta_output = output - iris_output
        weight_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        # Backpropogation (hidden layer)
        delta_hidden = np.transpose(weight_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weight_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(labels)
        bias_input_to_hidden += -learning_rate * delta_hidden

    print(f"Loss: {round((e_loss[0] / train_iris_data.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / train_iris_data.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# CHECK
# row = test_iris_data
# hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ row
# hidden = np.maximum(0, hidden_raw)
#
# output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
# output = np.maximum(0, output_raw)
#
# expected = test_iris_output.iloc[0]
# predict_output = output
#
# # print(predict_output)
# print(expected)
# print("For input: " + f"{row}" + " the prediction is: " + f"{predict_output[0]}" + ", expected: " + f"{expected}")
