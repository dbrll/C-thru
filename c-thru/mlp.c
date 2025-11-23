// Build with: gcc -shared -fPIC -O3 -o mlp.so mlp.c -lm

// That all we'll need!
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// for the RNG seed
#include <time.h> 
// to load and save the weights and biases
#include <string.h> 

// Network dimensions:
// input size (28x28 neurons)
// hidden layer (32)
// output classes (0-9)
#define INPUT 784
#define HIDDEN 32
#define OUTPUT 10

// Weight matrices and bias vectors
double W1[INPUT][HIDDEN];    // Input → Hidden
double b1[HIDDEN];           // Hidden layer biases

double W2[HIDDEN][OUTPUT];   // Hidden → Output
double b2[OUTPUT];           // Output layer biases

// Initialize weights with small random values
void init_weights() {

    srand(time(NULL));       // Seed RNG with current time

    // Random weights in [-0.01, 0.01]
    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HIDDEN; j++)
            W1[i][j] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;

    // Hidden layer biases start at zero
    for (int j = 0; j < HIDDEN; j++) b1[j] = 0.0;

    // Same initialization for hidden → output weights  
    for (int j = 0; j < HIDDEN; j++)
        for (int k = 0; k < OUTPUT; k++)
            W2[j][k] = (rand() / (double)RAND_MAX) * 0.02 - 0.01;

     // Output biases also start at zero      
    for (int k = 0; k < OUTPUT; k++) b2[k] = 0.0;
}

// Forward pass: compute hidden and output activations
// That means: if we load an image in the INPUT layer,
// what's the predicted value in the OUTPUT layer?
void forward(double *x, double *h1, double *h2) {

    // Hidden layer activations
    for (int j = 0; j < HIDDEN; j++) {
        double s = b1[j];
        for (int i = 0; i < INPUT; i++) s += x[i] * W1[i][j];
        h1[j] = tanh(s);         // Hidden activation (can be swapped with ReLU)
        //h1[j] = fmax(0.0, s);  // ReLU 
    }
    // Output layer activations
    for (int k = 0; k < OUTPUT; k++) {
        double s = b2[k];
        for (int j = 0; j < HIDDEN; j++) s += h1[j] * W2[j][k];
        h2[k] = tanh(s);        // Output activation
    }
}

// Backpropagation + gradient descent update
// That means: given the prediction error, compute how each weight
// contributed to it and adjust them to reduce future errors
void backward(double *x, double *y) {

    float lr = 0.01; // Learning rate
    double h1[HIDDEN], h2[OUTPUT];

    // Recompute forward pass
    forward(x, h1, h2);

    // The forward pass just gave us the predicted output value
    // How wrong is it compared to the actual label value? That's delta2
    double delta2[OUTPUT];
    for (int k = 0; k < OUTPUT; k++)
        delta2[k] = h2[k] - y[k];

    // Update weights W2 and biases b2 with delta2 to apply the error correction
    for (int j = 0; j < HIDDEN; j++)
        for (int k = 0; k < OUTPUT; k++)
            W2[j][k] -= lr * h1[j] * delta2[k];
    for (int k = 0; k < OUTPUT; k++) b2[k] -= lr * delta2[k];

    // Moving to the previous layer
    // delta1 is the hidden neuron’s share of the output error: 
    // the weighted sum of delta2, scaled by the activation function’s slope
    double delta1[HIDDEN];
    for (int j = 0; j < HIDDEN; j++) {
        double s = 0.0;
        for (int k = 0; k < OUTPUT; k++) s += delta2[k] * W2[j][k];
        delta1[j] = s * (1.0 - h1[j] * h1[j]);
    }

    // Update W1 and b1 with delta1
    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < HIDDEN; j++)
            W1[i][j] -= lr * x[i] * delta1[j];
    for (int j = 0; j < HIDDEN; j++) b1[j] -= lr * delta1[j];
}

// Train the whole dataset for several epochs
void train(int epochs, double *images, int *labels, int n_samples) {

    for (int e = 0; e < epochs; e++) {
        printf("Epoch %d/%d: ", e + 1, epochs);
        int correct = 0;
        double mse = 0.0;  // Mean Squared Error

        // For each epoch, load the image and its corresponding label
        // Then run a forward pass, compute the error, and backpropagate it

        for (int i = 0; i < n_samples; i++) {
            // Input vector (the image)
            double x[INPUT];  
            // One-hot label (the corresponding value)
            double y[OUTPUT] = {0};  

            // Load image pixels into the input layer
            for (int j = 0; j < INPUT; j++) 
                x[j] = images[i * INPUT + j];
            
            // One-hot encoding (load the label value in the output layer)
            y[labels[i]] = 1.0;

            // Activations between hidden and output layers
            double h1[HIDDEN], h2[OUTPUT];
            // Retrieving the activations values after a forward pass
            forward(x, h1, h2); 

            // Determine predicted class
            int pred = 0;
            for (int k = 1; k < OUTPUT; k++)
                if (h2[k] > h2[pred]) pred = k;

            // If the prediction is correct
            if (pred == labels[i]) correct++;

            // Accumulate MSE for each OUTPUT
            for (int k = 0; k < OUTPUT; k++) {
                mse += (y[k] - h2[k]) * (y[k] - h2[k]);
            }

            // Update weight and biases
            backward(x, y);
        }
        printf("accuracy = %.4f", (float)correct / n_samples);
        printf(", MSE = %.4f\n", mse / n_samples);  
    }
}

// Load weights from a serialized text buffer
void load_weights(const char *data) {
    char line[256];
    char type[10];
    int i, j;
    double val;

    char *str = strdup(data);
    char *ptr = str;
    char *line_start;

    while ((line_start = strsep(&ptr, "\n")) != NULL) {
        if (line_start[0] == '\0' || line_start[0] == '#') continue;

        int items = sscanf(line_start, "%s %d %d %lf", type, &i, &j, &val);
        if (items == 4) {
            if (strcmp(type, "W1") == 0) W1[i][j] = val;
            else if (strcmp(type, "W2") == 0) W2[i][j] = val;
        } else {
            items = sscanf(line_start, "%s %d %lf", type, &i, &val);
            if (items == 3) {
                if (strcmp(type, "b1") == 0) b1[i] = val;
                else if (strcmp(type, "b2") == 0) b2[i] = val;
            }
        }
    }
    free(str);
    printf("Weights loaded (W1, b1, W2, b2)\n");
}

// Serialize weights into a text buffer
void save_weights(char *buffer, int buffer_size) {
    char *ptr = buffer;
    int remaining = buffer_size;

    for (int i = 0; i < INPUT; i++) {
        for (int j = 0; j < HIDDEN; j++) {
            int written = snprintf(ptr, remaining, "W1 %d %d %.6f\n", i, j, W1[i][j]);
            if (written >= remaining) return;  // Buffer is full
            ptr += written;
            remaining -= written;
        }
    }
    for (int j = 0; j < HIDDEN; j++) {
        int written = snprintf(ptr, remaining, "b1 %d %.6f\n", j, b1[j]);
        if (written >= remaining) return;
        ptr += written;
        remaining -= written;
    }
    for (int j = 0; j < HIDDEN; j++) {
        for (int k = 0; k < OUTPUT; k++) {
            int written = snprintf(ptr, remaining, "W2 %d %d %.6f\n", j, k, W2[j][k]);
            if (written >= remaining) return;
            ptr += written;
            remaining -= written;
        }
    }
    for (int k = 0; k < OUTPUT; k++) {
        int written = snprintf(ptr, remaining, "b2 %d %.6f\n", k, b2[k]);
        if (written >= remaining) return;
        ptr += written;
        remaining -= written;
    }
    if (remaining > 0) {
        ptr[0] = '\0';
    }
}