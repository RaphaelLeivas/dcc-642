import numpy as np
np.random.seed(42)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pdb

def load_mnist():
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', as_frame=False, cache=True, version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Normalize the data
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test

def initialize_parameters(input_size, hidden_size, output_size):
    b1 = np.zeros(hidden_size)
    b2 = np.zeros(output_size)

    W1_n = input_size
    W1_m = hidden_size

    W2_n = hidden_size
    W2_m = output_size

    W1 = np.zeros((W1_n, W1_m))
    W2 = np.zeros((W2_n, W2_m))

    for i in range(W1_n):
        for j in range(W1_m):
            W1[i][j] = np.random.randn() * 0.01

    for i in range(W2_n):
        for j in range(W2_m):
            W2[i][j] = np.random.randn() * 0.01

    return (W1, b1, W2, b2)
    

# Activation functions
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# Forward pass
def forward(X, parameters):
    (W1, b1, W2, b2) = parameters
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    return (Z1, A1, Z2, A2)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)

# Backward pass
def backward(X, y, W2, Z1, A1, A2):
    n_samples = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    dA1 = np.dot(dZ2, W2.T)
    
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
    
    return dW1, db1, dW2, db2

def gradient_descent_step(X, y, parameters, learning_rate=0.01):
    (Z1, A1, Z2, A2) = forward(X, parameters)
    (W1, b1, W2, b2) = parameters

    # pdb.set_trace()

    # calcula a saida da rede
    y_hat = relu(Z2)

    l = cross_entropy_loss(y, y_hat)

    dW1, db1, dW2, db2 = backward(X, y, W2, Z1, A1, A2)

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    return (W1, b1, W2, b2), l

def accuracy(y_true, y_pred):
    correct = 0
    n, m = y_true.shape

    for i in range(n):
        current_prediction = np.argmax(y_pred[i])
        true_value = np.argmax(y_true[i])
        if current_prediction == true_value: correct += 1

    return correct / len(y_true)
    
    

    










