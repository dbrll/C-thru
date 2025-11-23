# This is a standalone C-thru implementation in pure Python + NumPy

import gzip
import pickle
import numpy as np

# Display
import os
import sys
# A bit hacky, but spares us from creating a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared')))
from visualize import live_network_visualization, draw_digit, visualize_weights


def load_mnist():

    """
    Load the MNIST dataset (images and labels) from a gzipped pickle file
    The function extracts the training and test sets, returning them as numpy arrays

    Return:
        X_train, Y_train (numpy arrays): Training data (images) and corresponding labels
        X_test, Y_test (numpy arrays): Test data (images) and corresponding labels
    """

    with gzip.open('../mnist.pkl.gz', 'rb') as f:
        train_set, test_set, _ = pickle.load(f, encoding='latin1')
    X_train, Y_train = train_set
    X_test, Y_test = test_set
    return X_train, Y_train


lr = 0.1                # Learning rate
epochs = 5              # Number of epochs to train the network

# Architecture:
# 784 input neurons (28x28 images)
# 32 hidden neurons
# 10 output neurons
layers = [784, 32, 10]

# Weight and biases (randomization with small values)
W = [np.random.randn(layers[i], layers[i+1]) * 0.05
     for i in range(len(layers)-1)]
b = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]


def to_one_hot(Y, num_classes=10):

    """
    Convert the labels to a one-hot encoded format.
    Example: If the label is 2, return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].

    Args:
        Y (numpy array): The labels as integers (0-9).
        num_classes (int): The number of output classes (default is 10 for MNIST).

    Returns:
        numpy array: One-hot encoded labels.
    """
    return np.eye(num_classes)[Y]


def forward(X):

    """
    Performs the forward pass of the neural network.
    This computes the activations of each layer, including the output layer
    with softmax activation

    Args:
        X (numpy array): The input data (batch of images)

    Returns:
        activations (list): List of activations for each layer
        z_values (list): List of z-values (raw inputs to each activation function)
    """

    activations = [X]       # Activations by layer
    z_values = []           # Z values before activation
    a = X                   # Current activation

    # For each layer of the network from input to output
    for i in range(len(W)):

        # z = raw value before activation function
        # z = previous activation * (W+b)
        z = a @ W[i] + b[i]

        if i == len(W)-1:
            # If output layer, apply a softmax activation to the Z values
            # softmax = exponential sum
            expz = np.exp(z - np.max(z, axis=1, keepdims=True))
            a = expz / np.sum(expz, axis=1, keepdims=True)
        else:
            # If hidden layer, use a tanh activation
            # where tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
            a = np.tanh(z)

        z_values.append(z)
        activations.append(a)
    return activations, z_values


def backward(X, y, activations, z_values, lr):
    
    """
    Perform backpropagation to compute gradients and update the weights and biases

    Args:
        X (numpy array): The input data for this batch
        y (numpy array): The one-hot encoded labels
        activations (list): The activations of each layer
        z_values (list): The raw pre-activation values for each layer
        lr (float): The learning rate for gradient descent
    """

    # How many training pictures
    m = X.shape[0]
    # How many layers in the network
    L = len(W)

    # Output error
    # = output layer activations - image label
    delta = activations[-1] - y

    # From the output layer back to the input layer
    for i in reversed(range(L)):
        # Weights gradient = previous layer activation * delta / nb images
        dW = activations[i].T @ delta / m
        # Bias gradient = deltas average
        db = np.mean(delta, axis=0, keepdims=True)

        # Weight correction: subtract dW * lr
        W[i] -= lr * dW
        # Same for biases
        b[i] -= lr * db

        # If not on the input layer, calculate delta for the previous layer
        if i != 0:
            delta = (delta @ W[i].T) * (1 - np.tanh(z_values[i-1])**2)


# MAIN
print("Loading MNIST dataset...")

X, Y = load_mnist()     # Load the training images and labels
Y = to_one_hot(Y)       # Convert labels to one-hot encoding

batch_size = 4          # Size of the mini-batches used during training
batch_size = 4          # Size of the mini-batches used during training
n_samples = X.shape[0]  # Number of samples in the dataset

# Store initial weights and biases (used for visualizations later)
initial_W = [w.copy() for w in W]
initial_b = [b.copy() for b in b]

print("Training...")
for epoch in range(epochs):
    # Shuffle the data to prevent the network from learning in a fixed order
    # This ensures the model sees the data in a random sequence during training
    indices = np.random.permutation(n_samples) # Randomly shuffle the dataset
    X = X[indices] # Shuffle input data
    Y = Y[indices] # Shuffle labels

    # Mini-batch gradient descent
    for i in range(0, n_samples, batch_size):       # Go through all the images, batch_size at a time
        Xb = X[i:i+batch_size].astype('float64')    # Get a batch of input images
        Yb = Y[i:i+batch_size]                      # Get corresponding labels
        activations, z_values = forward(Xb)         # Perform forward pass (predicition for each)
        backward(Xb, Yb, activations, z_values, lr) # Perform backward pass (gradient update)

    # Evaluate the model's accuracy and mean squared error (MSE) after each epoch
    activations_full, _ = forward(X)
    preds = np.argmax(activations_full[-1], axis=1)
    labels = np.argmax(Y, axis=1)
    acc = np.mean(preds == labels)
    mse = np.mean(np.square(activations_full[-1] - Y))
    print(f"Epoch {epoch+1}/{epochs}: accuracy = {acc:.4f}, MSE = {mse:.4f}")

# Visualize the trained weights after training
print("Layers: ", layers)
visualize_weights(W, layers)


def get_activations():

    """
    Return the current activations of the network
    This function is used by the live visualization to update the network state
    """

    global current_activations
    return current_activations


def recognize(canvas):

    """
    Recognize a digit from a 28x28 canvas (reshaped into a 1D vector of length 784)
    This function updates the current activations and triggers the visualization

    Args:
        canvas (numpy array): The input image (28x28) of the digit to recognize
    """

    global current_activations
    # Flatten the drawn digit (28x28px) into a vector for the input layer
    x = canvas.reshape(1, 784)
    # What's the prediction for this digit?
    activations, _ = forward(x)
    current_activations = activations


canvas_init = np.zeros((28, 28))
recognize(canvas_init)
draw_digit(recognize)
live_network_visualization(W, get_activations)
