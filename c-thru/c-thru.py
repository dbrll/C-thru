import os
import io
import sys
import gzip
import pickle
import numpy as np
from cffi import FFI

# Displays import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared')))
from visualize import live_network_visualization, draw_digit, visualize_weights


#
# FFI to the C MLP engine (mlp.c)
#

ffi = FFI()
ffi.cdef("""
void init_weights();
void forward(double *x, double *h1, double *h2);
void train(int epochs, double *images, int *labels, int n_samples);
void save_weights(char *buffer, int buffer_size);
void load_weights(const char *data);
""")
mlp = ffi.dlopen('./mlp.so')


#
# Handling weights and biases
#

WEIGHTS_FILE = "Wb.txt"
BUFFER_SIZE = 1024 * 1024  # 1 MB is enough to fit all the data (~500KB)

if not os.path.exists(WEIGHTS_FILE):
    print("Cannot find weights and biases (Wb.txt). Training the network...")

    # Load the MNIST data for the training
    with gzip.open('../mnist.pkl.gz', 'rb') as f:
        train_set, _, _ = pickle.load(f, encoding='latin1')
    X, y = train_set
    X = X.astype('float32')
    y = y.astype('int32')

    print(f"  {len(X)} images loaded. Training on 5 epochs...")
    images = ffi.new("double[]", X.flatten().tolist())
    labels = ffi.new("int[]", y.tolist())

    # Initializing weights
    mlp.init_weights()
    # Start the training
    mlp.train(5, images, labels, len(X))

    # Save weights to file after training
    buffer = ffi.new("char[]", BUFFER_SIZE)
    mlp.save_weights(buffer, BUFFER_SIZE)
    weights_data = ffi.string(buffer).decode('utf-8')

    with open(WEIGHTS_FILE, 'w') as f:
        f.write(weights_data)

    print(f"mnist.py: Training done, {WEIGHTS_FILE} saved")

else:
    print(f"{WEIGHTS_FILE} found → Using trained weights")

    with open(WEIGHTS_FILE, 'r') as f:
        weights_data = f.read()

    c_data = ffi.new("char[]", weights_data.encode('utf-8'))
    mlp.load_weights(c_data)


#
# Predicting a drawn digit
#

current_activations = None


def recognize(canvas):
    global current_activations
    x = canvas.flatten().astype('float32')
    x_c = ffi.new("double[784]", x.tolist())
    h1_c = ffi.new("double[32]")
    h2_c = ffi.new("double[10]")
    mlp.forward(x_c, h1_c, h2_c)
    h1 = np.array([h1_c[i] for i in range(layer_size[1])])
    h2 = np.array([h2_c[i] for i in range(layer_size[2])])

    current_activations = [
        x.reshape(1, 784),   # (1, 784)
        np.array(h1, dtype=float).reshape(1, layer_size[1]),   # (1, 16)
        np.array(h2, dtype=float).reshape(1, layer_size[2])    # (1, 10)
    ]


def get_activations():
    # Returns the activation for the visualization
    global current_activations
    return current_activations


def weight_file_to_array(filename):

    # Reads a weight file written by the C MLP and rebuilds
    # numpy arrays: W[0], W[1] to visualize them

    # Temporary structures before converting to arrays
    W_tmp = { "W1": {}, "W2": {} }

    # We also detect max indices to infer matrix sizes
    max_i = { "W1": 0, "W2": 0 }
    max_j = { "W1": 0, "W2": 0 }

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Format:  W1 i j val
            if parts[0] in ("W1", "W2"):
                name, i, j, val = parts
                i = int(i)
                j = int(j)
                val = float(val)

                W_tmp[name][(i, j)] = val

                if i > max_i[name]: max_i[name] = i
                if j > max_j[name]: max_j[name] = j

    # Now build numpy arrays with correct sizes
    W1 = np.zeros((max_i["W1"] + 1, max_j["W1"] + 1))
    W2 = np.zeros((max_i["W2"] + 1, max_j["W2"] + 1))

    # Fill arrays
    for (i, j), val in W_tmp["W1"].items():
        W1[i, j] = val
    for (i, j), val in W_tmp["W2"].items():
        W2[i, j] = val

    # Return clean lists for convenience
    W = [W1, W2]

    return W


if __name__ == "__main__":

    layer_size = [784, 32, 10]

    # Fictive weights to have the proper shapes before anything
    W = [
        np.zeros((layer_size[0], layer_size[1])),  # W1
        np.zeros((layer_size[1], layer_size[2]))   # W2
    ]
    canvas = np.zeros((28, 28))

    # Visualize the weights
    W = weight_file_to_array(WEIGHTS_FILE)
    visualize_weights(W, layer_size)

    # Start the digit drawing and live visualization displays
    recognize(canvas)
    draw_digit(recognize)
    live_network_visualization(W, get_activations, interval=50)