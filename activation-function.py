import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

def swish(x):
    return x * sigmoid(x)

# Activation functions dictionary
activation_functions = {
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'ReLU': relu,
    'Leaky ReLU': leaky_relu,
    'ELU': elu,
    'SELU': selu,
    'Softmax': softmax,
    'Swish': swish
}

# Create a small dataset
x = np.linspace(-10, 10, 100)

# Create subplots for each activation function
rows, cols = 4, 2
fig, axs = plt.subplots(rows, cols, figsize=(12, 16))

for i, (name, activation_func) in enumerate(activation_functions.items()):
    row = i // cols
    col = i % cols
    y = activation_func(x)
    axs[row, col].plot(x, y, color='red')
    axs[row, col].set_title(name)
    axs[row, col].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
