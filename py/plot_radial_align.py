import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Plot radial solution curves from data file.")
parser.add_argument("data_files", type=str, nargs='+', help="Path to the data file with radial solution data")

# Parse the arguments
args = parser.parse_args()

starting_values = []
curves = [] 
labels = [] 

# Load the data from the specified file
for data_file in args.data_files:
    with open(data_file, 'r') as f:
        label = f.readline().strip()
    data = np.loadtxt(data_file, skiprows=1)
    radii = data[:, 0]
    values = data[:, 1:]
    
    labels.append(label)
    curves.append((radii, values))
    starting_values.append(values[0, 0])

alignment_value = starting_values[-1]

for idx, (radii, values) in enumerate(curves):
    # Normalize the starting point of each curve
    shift = alignment_value - starting_values[idx]
    normalized_values = values + shift

    # Plot the curves
    for curve_idx in range(normalized_values.shape[1]):
        plt.plot(radii, normalized_values[:, curve_idx], label=f'{labels[idx]}')

plt.xlim(0, 10)
plt.xlabel("Radius")
plt.ylabel("Solution Value")
plt.title("Radial Distribution of Solution for Different Meshes")
plt.legend()
plt.grid()
plt.show()

