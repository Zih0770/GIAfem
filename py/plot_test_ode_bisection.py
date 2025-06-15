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
        data = np.loadtxt(f'solver_{s_type}_N.dat')
        target_error, dt, N_calls, actual_error = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        
        plt.loglog(actual_error, N_calls, '-o', label=label)
    except FileNotFoundError:
        print(f"Data file for solver {label} (solver_{s_type}_N.dat) not found, skipping...")

plt.xlabel('L2 Error at $t=30$')
plt.ylabel('Number of Mult calls (N)')
plt.title('ODE Solver Comparison ($t_{final}=30$)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('solver_comparison_N_vs_L2_bisection.png', dpi=300)
plt.show()
