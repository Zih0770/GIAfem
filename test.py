import glob
import numpy as np
import matplotlib.pyplot as plt

filenames = sorted(glob.glob("output_*.dat"))

plt.figure(figsize=(10, 4))
for fname in filenames:
    t, comp_time, err = np.loadtxt(fname, comments="#", unpack=True)
    label = fname.replace(".dat","").replace("output_", "")
    plt.semilogy(t, err, marker="o", markersize=1, linestyle="-", label=label)
    plt.text(t[-1], err[-1], label, fontsize=10, verticalalignment='center')

plt.xlabel("Time $t$")
#plt.ylabel("Error $||m-m_{ex}||_2$")
plt.ylabel("$||u||_2$")
#plt.ylim(top=1e-2, bottom=1e-16)
plt.title("Evolution of $L_2$ norm vs. Time")
plt.grid(True)
#plt.legend(loc="best")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
for fname in filenames:
    t, comp_time, err = np.loadtxt(fname, comments="#", unpack=True)
    label = fname.replace(".dat", "").replace("output_", "")
    plt.plot(t, comp_time, marker="o", markersize=1, linestyle="-", label=label)
    plt.text(t[-1], comp_time[-1], label, fontsize=10, verticalalignment='center')

plt.xlabel("Time $t$")
plt.ylabel("Computational Time [s]")
plt.ylim(bottom=0.0)
plt.title("Computational Time vs. Physical Time")
plt.grid(True)
#plt.legend(loc="best")
plt.tight_layout()

plt.show()


