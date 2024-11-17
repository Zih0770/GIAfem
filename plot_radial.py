import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Plot radial solution curves from data file.")
parser.add_argument("data_file", type=str, help="Path to the data file with radial solution data")

# Parse the arguments
args = parser.parse_args()

# Load the data from the specified file
data = np.loadtxt(args.data_file, skiprows=1)
radii = data[:, 0]
values = data[:, 1:]

# Plot the curves
for i in range(values.shape[1]):
    plt.plot(radii, values[:, i], label=f'Mesh {i+1}')

plt.xlabel("Radius")
plt.ylabel("Solution Value")
plt.title("Radial Distribution of Solution for Different Meshes")
plt.legend()
plt.grid()
plt.show()

