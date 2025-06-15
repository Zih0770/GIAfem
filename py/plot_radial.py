import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Plot radial solution curves from data file.")
parser.add_argument("data_files", type=str, nargs='+', help="Path to the data file with radial solution data")

# Parse the arguments
args = parser.parse_args()

# Load the data from the specified file
for data_file in args.data_files:
    with open(data_file, 'r') as f:
        label = f.readline().strip()
    data = np.loadtxt(data_file, skiprows=1)
    radii = data[:, 0]
    values = data[:, 1:]

    # Plot the curves
    for curve_idx in range(values.shape[1]):
        plt.plot(radii, values[:, curve_idx], label=f'{label}')

plt.xlim(0, 10)
plt.xlabel("Radius")
plt.ylabel("Solution Value")
plt.title("Radial Distribution of Solution for Different Meshes")
plt.legend()
plt.grid()
plt.show()

