# C-thru

C-thru is a teaching tool designed to show what’s happening inside a neural
network, **without hiding anything behind layers of abstraction**.

### What makes C-thru special?

- **A tiny C engine, simple and transparent** The core of the network is written
  in plain ISO C (about 200 lines of code) heavily commented and relying only on
  standard libraries: `stdio.h`, `stdlib.h`, and `math.h`. No classes. No
  framework. No magic. No hidden black boxes. No dependencies.

- **Human-readable parameters** All weights and biases the network learns are
  saved in a plain text file that can be read, inspected, or even modified by
  hand.

- **Live visualization in Python** The Python part first displays the network’s
  weights as images, then shows how the neural network changes its prediction as
  digits are drawn.

A pure Python/NumPy implementation is also included, using the same
visualization tools, for those who prefer a shorter and more abstracted
implementation.

[![Thumbnail](https://github.com/user-attachments/assets/abcc46e8-33ae-4d05-98de-f37246961d5f)](https://github.com/user-attachments/assets/a88e788f-e54f-4e26-b51c-d41f6344d8be)

### Running it

C-thru only requires :

- Python with NumPy and Matplotlib
- a C compiler (gcc, clang)

To run C-thru:

```sh
cd c-thru
gcc -shared -fPIC -O3 -o mlp.so mlp.c -lm  # build the engine
python c-thru.py                           # train + live display
```

After the training is done, weights and biases will be saved to `Wb.txt`, the
weights will be displayed, and the live visualization will start.
