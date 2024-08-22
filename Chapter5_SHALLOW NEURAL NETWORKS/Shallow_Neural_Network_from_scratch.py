import numpy as np



# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Output data
y = np.array([[0], [1], [1], [0]])

# Seed for random weight initialization
np.random.seed(1)

# Initialize weights randomly with mean 0
weights_0 = 2 * np.random.random((2, 3)) - 1
weights_1 = 2 * np.random.random((3, 1)) - 1


# Training loop
for epoch in range(60000):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, weights_0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1))

    # Back propagation
    layer_2_delta = (layer_2 - y) * sigmoid_derivative(layer_2)
    layer_1_delta = layer_2_delta.dot(weights_1.T) * sigmoid_derivative(layer_1)

    # Update weights
    weights_1 -= layer_1.T.dot(layer_2_delta)
    weights_0 -= layer_0.T.dot(layer_1_delta)

# Print final weights
print("Final weights after training:")
print("Weights between input and hidden layer:")
print(weights_0)
print("Weights between hidden and output layer:")
print(weights_1)


