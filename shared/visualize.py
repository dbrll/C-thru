import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.collections import LineCollection


# Live network display
# ====================

def live_network_visualization(W, activations_provider, interval=100):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()

    activations = activations_provider()
    # --- Positions des neurones ---
    layer_x = np.linspace(0, 1, len(activations))
    node_positions = []
    for i, Wmat in enumerate(W):
        size = Wmat.shape[0]
        ypos = np.linspace(0, 1, size)
        xpos = np.ones(size) * layer_x[i]
        node_positions.append(np.column_stack([xpos, ypos]))
    # Couche de sortie
    output_size = W[-1].shape[1]
    output_xpos = np.ones(output_size) * layer_x[-1]
    output_ypos = np.linspace(0, 1, output_size)
    node_positions.append(np.column_stack([output_xpos, output_ypos]))

    # --- Create segments (connections) once ---
    segments = []
    for li, Wmat in enumerate(W):
        pts1 = node_positions[li]
        pts2 = node_positions[li + 1]
        for a in range(pts1.shape[0]):
            for b in range(pts2.shape[0]):
                segments.append([pts1[a], pts2[b]])
    segments = np.array(segments)

    # --- LineCollection ---
    lc = LineCollection(segments, linewidths=0.5, colors='gray', alpha=0.6)
    ax.add_collection(lc)

    # --- Scatter ---
    scatters = []
    for i, pos in enumerate(node_positions):
        s = 40 if i == 0 else 100
        scat = ax.scatter(pos[:, 0], pos[:, 1], s=s, c='lightgray', zorder=5)
        scatters.append(scat)

    # --- Layer labels ---
    n_input = W[0].shape[0]          # 784
    n_hidden = W[0].shape[1]         # 32
    n_output = W[-1].shape[1]        # 10
    layer_names = [f"Input\n({n_input} neurons)", f"Hidden\n({n_hidden} neurons)", f"Output\n({n_output} neurons)"]
    for i, (pos, name) in enumerate(zip(node_positions, layer_names)):
        ax.text(np.mean(pos[:, 0]), -0.05, name, ha='center', va='top', fontsize=12)

    # --- Output neurons labels : "9   0.000" ---
    output_labels = []
    for i in range(output_size):
        y = node_positions[2][i, 1]
        digit = 9 - i  # 9 en haut, 0 en bas
        label = ax.text(1.05, y, f"{digit}   0.000", ha='left', va='center',
                        fontsize=11, fontweight='bold', color='black')
        output_labels.append(label)

    ax.set_xlim(-0.1, 1.25)  # space for the labels
    ax.set_ylim(-0.1, 1.1)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label('Activation')

    # --- How many parameters? ---
    sizes = []
    for i in range(len(activations)):
        size = activations[i].shape[1]
        sizes.append(size)
    params = 0
    for i in range(len(sizes) - 1):
        poids = sizes[i] * sizes[i + 1]
        biais = sizes[i + 1] 
        params += poids + biais

    ax.text(0.44,-0.1, f"Total parameters: {params:,}", transform=ax.transAxes,
           ha='center', va="top", fontsize=11)

    fig.canvas.manager.set_window_title("Neural Network")

    # ---Animation update ---
    def update(frame):
        activations = activations_provider()
        if not activations or len(activations) < 3:
            return [lc] + scatters + output_labels

        input_act = activations[0].flatten()
        hidden_act = activations[1].flatten()
        output_act = activations[2].flatten()

        all_acts = np.concatenate([input_act, hidden_act, output_act])
        if all_acts.max() > all_acts.min():
            norm = mcolors.Normalize(vmin=all_acts.min(), vmax=all_acts.max())
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

        # --- Neurons ---
        active = input_act >= 1
        # colors_in = np.ones((len(input_act), 4)) * 0.8
        colors_in = np.full((len(input_act), 4), plt.cm.viridis(0))  # ← Dark blue by default
        if np.any(active):
            vals = input_act[active]
            colors_in[active] = plt.cm.viridis(norm(vals))
        scatters[0].set_facecolor(colors_in)
        scatters[0].set_sizes(np.where(active, 40, 20))

        for i, act in enumerate([hidden_act, output_act], 1):
            colors = plt.cm.viridis(norm(act))
            scatters[i].set_facecolor(colors)

        # --- Connections ---
        linewidths = []
        colors = []

        n_input = len(input_act)
        n_hidden = len(hidden_act)
        n_output = len(output_act)

        # Input → Hidden : only if input_act[a] >= 1
        for a in range(n_input):
            if input_act[a] >= 1:
                for b in range(n_hidden):
                    weight = abs(W[0][a, b])
                    linewidths.append(0.1 + 2.0 * weight * 1.0)
                    colors.append(plt.cm.viridis(0.5))  # bleu foncé fixe
            else:
                for b in range(n_hidden):
                    linewidths.append(0.0)
                    colors.append((0, 0, 0, 0))

        # Hidden → Output
        for a in range(n_hidden):
            for b in range(n_output):
                weight = abs(W[1][a, b])
                act_val = hidden_act[a]
                linewidths.append(0.3 + 2.0 * weight * act_val)
                colors.append(plt.cm.viridis(norm(act_val)))

        lc.set_linewidths(linewidths)
        lc.set_colors(colors)

        # --- Update the output value : "digit   value" ---
        for i, val in enumerate(output_act):
            digit = i
            norm = mcolors.Normalize(vmin=output_act.min(), vmax=output_act.max())
            color = plt.cm.viridis(norm(val))  # Appliquer la palette Viridis
            output_labels[i].set_text(f"{digit}   {val:.4f}")
            output_labels[i].set_color(color)

        # --- Colorbar ---
        sm.norm = norm
        sm.set_array(all_acts)
        cbar.update_normal(sm)

        return [lc] + scatters + output_labels

    # --- Animation ---
    ani = animation.FuncAnimation(
        fig, update, interval=interval, blit=False, cache_frame_data=False
    )
    plt.show()
    return ani


# Digit drawing window
# ====================

def draw_digit(callback):
    import matplotlib.pyplot as plt
    import numpy as np

    canvas = np.zeros((28, 28))

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Draw a digit (right click to clear)")

    img = ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

    drawing = False

    def on_press(event):
        nonlocal drawing
        drawing = True
        if event.button == 3:  # Right click
            reset_canvas()
            callback(canvas)

    def on_release(event):
        nonlocal drawing
        drawing = False

    def on_move(event):
        nonlocal canvas
        if not drawing:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.ydata), int(event.xdata)
        canvas[max(0, x-1):x+2, max(0, y-1):y+2] = 1.0
        img.set_data(canvas)
        fig.canvas.draw_idle()
        callback(canvas)

    def reset_canvas():
        nonlocal canvas
        canvas = np.zeros((28, 28))  # Blank the drawing canvas
        img.set_data(canvas)
        fig.canvas.draw_idle()
        callback(canvas)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    def on_close(event):
        sys.exit()

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show(block=False)

#
# Weights display
# ===============


def visualize_weights(W, layers, img_size=(28, 28)):
    """
    Visualize the weights from each hidden layer to the next layer.
    Each neuron in a hidden layer is reshaped as img_size (default 28x28)
    and displayed in a grid. Works for any number of neurons.
    """
    num_layers = len(layers)

    # Loop over each hidden layer except the output layer
    for layer_idx in range(num_layers - 2):
        weight_matrix = W[layer_idx]  # shape: (n_input, n_neurons)
        num_neurons = layers[layer_idx + 1]

        # Determine grid size
        grid_size = math.ceil(math.sqrt(num_neurons))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()  # flatten in case of multiple rows/cols

        for neuron_idx in range(grid_size * grid_size):
            ax = axes[neuron_idx]
            ax.axis("off")  # remove ticks by default

            if neuron_idx < num_neurons:
                neuron_weights = weight_matrix[:, neuron_idx]
                img = neuron_weights.reshape(img_size)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Neuron {neuron_idx + 1}", fontsize=8)

        fig.canvas.manager.set_window_title(f"Trained weights from layer {layer_idx} → {layer_idx + 1}")
        plt.tight_layout()
        plt.show(block=False)
