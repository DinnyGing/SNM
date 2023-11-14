import math

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import numpy as np


def euclidean_distance(weights, inputs):
    d = 0
    for i in range(len(inputs)):
        d += math.pow((weights[i] - inputs[i]), 2)
    return d


def gaussian_function(x):
    return np.exp(-x**2 / (2 * sigma**2))


def get_distances(data, som_weights):
    distances = []
    for i in range(len(som_weights)):
        distances.append(euclidean_distance(som_weights[i], data))
    return np.array(distances)


def find_best_matching_unit(data, som_weights):
    return np.argmin(get_distances(data, som_weights))


def update_weights(weights, input):
    n = find_best_matching_unit(input, weights)
    calc_weights = np.copy(weights)
    calc_weights[n] += learning_rate * gaussian_function(euclidean_distance(input, weights[n]))
    return calc_weights


def train(weights, inputs):
    calc_weights = weights
    for i in range(len(inputs)):
        calc_weights = update_weights(calc_weights, inputs[i])
    return calc_weights

def quantization_error(data, som_weights):
    quantization_error = 0
    for input_data in data:
        best_matching_unit_index = find_best_matching_unit(input_data, som_weights)
        quantization_error += gaussian_function(euclidean_distance(input_data, som_weights[best_matching_unit_index]))
    return quantization_error / len(data)


def topographic_error(data, som_weights):
    topographic_error = 0
    for i, input_data in enumerate(data):
        best_matching_unit_index = find_best_matching_unit(input_data, som_weights)

        # Знайдемо індекс найближчого сусіднього нейрона до best_matching_unit
        neighbors = find_neighbors(best_matching_unit_index, som_weights)

        # Перевіримо, чи знаходяться сусіди поруч (в топологічному сенсі)
        topographic_error += int(np.any(np.linalg.norm(som_weights[neighbors] - som_weights[best_matching_unit_index], axis=1) > 1))

    return topographic_error / len(data)

def find_neighbors(unit_index, som_weights):
    distances = []
    for i in range(len(som_weights)):
        distances.append(euclidean_distance(som_weights[i], som_weights[unit_index]))
    neighbors = np.where(np.array(distances) > 0)[0]

    return neighbors



X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Стандартизація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

input_layer = 2
output_layer = 9
sigma = 3
trained_weights = np.random.normal(-0.5, 0.5, (output_layer, input_layer))
epochs = 100
learning_rate = 0.017
for e in range(epochs):
    trained_weights = train(trained_weights, X)
    quantization_err = quantization_error(X, trained_weights)
    print("Quantization Error:", quantization_err)
    topographic_err = topographic_error(X, trained_weights)
    print("Topographic Error:", topographic_err)
