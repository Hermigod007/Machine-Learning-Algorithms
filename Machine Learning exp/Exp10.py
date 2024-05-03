import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

data_points = np.linspace(0, 10, 100)
function_values = np.exp(-0.1 * data_points)

plt.xlabel("x")
plt.ylabel("f(x)=e^-0.1x")
plt.title("Overall Dataset")
plt.plot(data_points, function_values)

kf = KFold(n_splits=5)
train_indices = np.empty((5, 80), dtype=int)
test_indices = np.empty((5, 20), dtype=int)
plt.figure(figsize=(15, 15))

for i, (train_index, test_index) in enumerate(kf.split(data_points, function_values)):
    train_indices[i, :] = train_index.T
    test_indices[i, :] = test_index.T

    train_data, test_data = data_points[train_index], data_points[test_index]
    train_values, test_values = function_values[train_index], function_values[test_index]
    plt.subplot(320 + 1 + i)
    plt.plot(train_data, train_values, 'r')
    plt.plot(test_data, test_values, 'b')
    plt.xlabel("x")
    plt.ylabel("f(x)=e^-0.1x")
    plt.title('Train and Test Data Sets')
    plt.legend(['train', 'test'])
plt.show()

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def forward_pass(x, weight, bias):
    a1 = x * weight[0] + bias[0]
    z1 = sigmoid(a1)
    a2 = z1 * weight[1] + bias[1]
    z2 = sigmoid(a2)
    return z1, z2

def gradient(x, y, weight, z1, z2):
    temp = -1 * (y - z2) * (z2) * (1 - z2)
    grad_b1 = temp * weight[1] * (z1) * (1 - z1)
    grad_w1 = grad_b1 * x
    grad_b2 = temp
    grad_w2 = temp * z1
    return grad_w1, grad_b1, grad_w2, grad_b2

weight = np.array([0, 0])
bias = np.array([0.01, 0.01])
epochs = 5
learning_rate = 1

for epoch in range(epochs):
    train_data, train_values = data_points[train_indices[epoch, :]], function_values[train_indices[epoch, :]]
    for i in range(80):
        z1, z2 = forward_pass(train_data[i], weight=weight, bias=bias)
        grad_w1, grad_b1, grad_w2, grad_b2 = gradient(train_data[i], train_values[i], weight=weight, z1=z1, z2=z2)
        weight[0] -= learning_rate * grad_w1
        bias[0] -= learning_rate * grad_b1
        weight[1] -= learning_rate * grad_w2
        bias[1] -= learning_rate * grad_b2

    test_data, test_values = data_points[test_indices[epoch, :]], function_values[test_indices[epoch, :]]
    predicted_values = np.empty(test_values.shape)
    for i in range(20):
        _, predicted_values[i] = forward_pass(test_data[i], weight=weight, bias=bias)

    rmse = mean_squared_error(y_true=test_values, y_pred=predicted_values)
    data = {'Epoch': epoch + 1, 'Weights': weight, 'Biases': bias, 'RMSE': rmse}
    df = pd.DataFrame(data)
    print(df)