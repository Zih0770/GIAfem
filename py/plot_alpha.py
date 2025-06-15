import numpy as np
import matplotlib.pyplot as plt

solver_labels = {
    1: "Forward Euler",
    2: "RK2",
    3: "RK3",
    4: "RK4",
    21: "Backward Euler",
    32: "Implicit Midpoint",
    33: "SDIRK23",
    99: "BaileySolver"
}

plt.figure(figsize=(10, 7))

for s_type, label in solver_labels.items():
    try:
        data = np.loadtxt(f'solver_{s_type}_alpha.dat')
        alpha, dt, N_calls, actual_error = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        plt.loglog(alpha, N_calls, '-o', label=label)
    except FileNotFoundError:
        print(f"File solver_{s_type}_alpha.dat not found, skipping...")

plt.xlabel(r'Forcing parameter $\alpha$')
plt.ylabel('Number of Mult calls (N)')
plt.title('N vs. alpha (L2 Error = 1e-6)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('N_vs_alpha.png', dpi=300)
plt.show()

