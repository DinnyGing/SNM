import idx2numpy
import numpy as np


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
        for j in range(len(inputs_1.T)):
            inputs_1.T[j] += calc_bias_0_1[j]
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, calc_weights_1_2.T)
        for j in range(len(inputs_2.T)):
            inputs_2.T[j] += calc_bias_1_2[j]
        outputs_2 = fun_l2(inputs_2)

        batch_labels_expected = [expect(label) for label in batch_labels]
        error_layer_2 = outputs_2 - batch_labels_expected
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        moving_average_squared_1_2 = rho * moving_average_squared_1_2 + (1 - rho) * (weights_delta_layer_2.T ** 2)
        calc_weights_1_2 -= (learning_rate / (np.sqrt(moving_average_squared_1_2 + epsilon))) * (outputs_1 @ weights_delta_layer_2).T
        calc_bias_1_2 -= (learning_rate / (np.sqrt(np.sum(weights_delta_layer_2, axis=0) + epsilon)))

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)

        moving_average_squared_0_1 = rho * moving_average_squared_0_1 + (1 - rho) * (weights_delta_layer_1 ** 2)
        calc_weights_0_1 -= (learning_rate / (np.sqrt(moving_average_squared_0_1 + epsilon))) * (batch_input @ weights_delta_layer_1).T
        calc_bias_0_1 -= (learning_rate / (np.sqrt(np.sum(weights_delta_layer_1, axis=0) + epsilon)))

    return calc_weights_0_1, calc_weights_1_2, calc_bias_0_1, calc_bias_1_2



def SGD(X, y):
    calc_weights_1_2 = []
    calc_weights_0_1 = []
    calc_bias_1_2 = []
    calc_bias_0_1 = []
    for i in range(0, len(X), batch_size):
        # Forward pass
        batch_input = X[i:i+batch_size].reshape(-1, input_layer_count)
        batch_labels = y[i:i+batch_size]

        inputs_1 = np.dot(batch_input, weights_0_1[i:i+batch_size].T)
        for i in range(len(inputs_1.T)):
            inputs_1.T[i] += bias_0_1[i:i+batch_size][i]
        outputs_1 = fun_l1(inputs_1)

        inputs_2 = np.dot(outputs_1, weights_1_2[i:i+batch_size].T)
        for i in range(len(inputs_2.T)):
            inputs_2.T[i] += bias_1_2[i:i+batch_size][i]
        outputs_2 = fun_l2(inputs_2)

        # Calculate the error
        batch_labels_expected = []
        for i in range(len(batch_labels)):
            batch_labels_expected.append(expect(batch_labels))
        batch_labels_expected = np.array(batch_labels_expected)
        error_layer_2 = outputs_2 - batch_labels_expected
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)
        calc_weights_1_2.append(weights_1_2[i:i+batch_size] - learning_rate * (outputs_1 @ weights_delta_layer_2).T)
        calc_bias_1_2.append(bias_1_2[i:i+batch_size] - learning_rate * weights_delta_layer_2)

        error_layer_1 = weights_delta_layer_2.dot(weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)
        calc_weights_0_1.append(weights_0_1[i:i+batch_size] - learning_rate * (batch_input @ weights_delta_layer_1).T)
        calc_bias_0_1.append(bias_0_1[i:i+batch_size] - learning_rate * weights_delta_layer_1)
    return calc_weights_0_1, calc_weights_1_2, \
        calc_bias_0_1[0], calc_bias_1_2[0]

def adam_optimizer(X, y):
    m = len(X)
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
        batch_input = X[i:i+batch_size].reshape(-1, input_layer_count)
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
        error_layer_2 = outputs_2 - batch_labels_expected
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        m_t_1_2_batch = np.sum(weights_delta_layer_2, axis=0)
        m_t_1_2_batch = m_t_1_2_batch.reshape((output_layer_count,))
        m_t_1_2 = beta1 * m_t_1_2 + (1 - beta1) * m_t_1_2_batch

        v_t_1_2 = beta2 * v_t_1_2 + (1 - beta2) * (weights_delta_layer_2 ** 2)

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)
        m_t_0_1 = beta1 * m_t_0_1 + (1 - beta1) * weights_delta_layer_1
        v_t_0_1 = beta2 * v_t_0_1 + (1 - beta2) * (weights_delta_layer_1 ** 2)

        m_t_hat_1_2 = m_t_1_2 / (1 - beta1 ** t)
        v_t_hat_1_2 = v_t_1_2 / (1 - beta2 ** t)

        m_t_hat_0_1 = m_t_0_1 / (1 - beta1 ** t)
        v_t_hat_0_1 = v_t_0_1 / (1 - beta2 ** t)

        calc_weights_1_2 -= learning_rate * (m_t_hat_1_2 / (np.sqrt(v_t_hat_1_2) + epsilon)).T
        calc_bias_1_2 -= learning_rate * np.mean(m_t_1_2_batch, axis=0)

        calc_weights_0_1 -= learning_rate * (m_t_hat_0_1 / (np.sqrt(v_t_hat_0_1) + epsilon)).T
        calc_bias_0_1 -= learning_rate * np.mean(weights_delta_layer_1, axis=0)

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
        batch_input = X[i:i+batch_size].reshape(-1, input_layer_count)
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
        error_layer_2 = outputs_2 - batch_labels_expected
        weights_delta_layer_2 = error_layer_2 * fun_back_l2(outputs_2)

        historical_gradient_squared_1_2 += (weights_delta_layer_2.T ** 2)
        calc_weights_1_2 -= (learning_rate / (np.sqrt(historical_gradient_squared_1_2 + epsilon))) * (outputs_1 @ weights_delta_layer_2).T
        calc_bias_1_2 -= (learning_rate / (np.sqrt(np.sum(weights_delta_layer_2, axis=0) + epsilon)))

        error_layer_1 = weights_delta_layer_2.dot(calc_weights_1_2)
        weights_delta_layer_1 = error_layer_1 * fun_back_l1(outputs_1)

        historical_gradient_squared_0_1 += (weights_delta_layer_1 ** 2)
        calc_weights_0_1 -= (learning_rate / (np.sqrt(historical_gradient_squared_0_1 + epsilon))) * (batch_input @ weights_delta_layer_1).T
        calc_bias_0_1 -= (learning_rate / (np.sqrt(np.sum(weights_delta_layer_1, axis=0) + epsilon)))

    return calc_weights_0_1, calc_weights_1_2, calc_bias_0_1, calc_bias_1_2




def train(inputs, expected_predict):
    outputs_1, outputs_2 = forward_propagation(inputs)
    return back_propagation(outputs_1, outputs_2, inputs, expected_predict)


def test(inputs, outputs):
    accuracy = 0
    for index in range(len(inputs)):
        labels = inputs[index].reshape(1, -1)[0]
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

train_data = np.array(idx2numpy.convert_from_file("train-images.idx3-ubyte"))
train_labels = np.array(idx2numpy.convert_from_file("train-labels.idx1-ubyte"))
test_data = np.array(idx2numpy.convert_from_file("t10k-images.idx3-ubyte"))
test_labels = np.array(idx2numpy.convert_from_file("t10k-labels.idx1-ubyte"))

hide_layer_count = 128
input_layer_count = len(train_data[0]) * len(test_data[0][0])
output_layer_count = 10
weights_0_1 = np.random.normal(-0.5, 0.5, (hide_layer_count, input_layer_count))
weights_1_2 = np.random.normal(-0.5, 0.5 ** -0.5, (output_layer_count, hide_layer_count))

bias_0_1 = np.zeros(hide_layer_count)
bias_1_2 = np.zeros(output_layer_count)

epochs = 5
learning_rate = 0.07
batch_size = 32
theta = None
q1 = 0
q2 = 0
fun_l1 = tanh
fun_l2 = sigmoid
fun_back_l1 = tanh_backward
fun_back_l2 = sigmoid_backward
train_data = train_data.reshape(len(train_data), input_layer_count)

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for index in range(len(train_data)):
        labels = train_data[index].reshape(1, -1)[0]
        expected = expect(train_labels[index])
        # weights_0_1, weights_1_2, \
        #     bias_0_1, bias_1_2 = rmsprop_optimizer(np.array(labels), expected)
        inputs_.append(np.array(labels))
        correct_predictions.append(np.array(expected))

    # here test
    weights_0_1, weights_1_2, \
        bias_0_1, bias_1_2 = adam_optimizer(np.array(train_data), train_labels)
    #here end test
    train_loss = categorical_crossentropy(np.array(correct_predictions),
                                          predict(np.array(inputs_).T).T)

    progress = 100 * e/float(epochs)
    if progress % 10 == 0:
        progress = str(progress)[:4]
        print("\rProgress: " + progress + " Training loss: ")
        print(train_loss)

print("Accuracy: " + str(test(test_data, test_labels)))
