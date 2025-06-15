import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

labels = {1:'FE', 2:'RK2', 3:'RK3', 4:'RK4', 21:'BE', 99:'Bailey'}

# ---- choose a width that matches your LaTeX \textwidth (≈6.5in),
# ---- and a height that keeps the aspect reasonable.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

# 1) left: frequency data
for fname in sorted(glob.glob('solver_*_frequency_1e8.dat'),
                    key=lambda fn: int(re.search(r'solver_(\d+)_', fn).group(1))):
    if os.path.getsize(fname) == 0:
        continue
    m = re.search(r'solver_(\d+)_frequency', fname)
    if not m:
        continue
    st = int(m.group(1))
    try:
        data = np.loadtxt(fname, comments='#')
    except ValueError:
        continue
    if data.ndim != 2 or data.shape[1] < 4:
        continue
    omega_tau, dt, calls, err = data.T
    ax1.loglog(omega_tau, calls, 'o-', label=labels.get(st, str(st)))

ax1.set_xlabel(r'$\omega\tau$')
ax1.set_title(r'$N$ vs $\omega\tau$')
ax1.grid(True, which='both', ls='--', alpha=0.5)
#ax1.legend(title='Solver', loc='best')
ax1.tick_params(direction='in', which='both', top=True, right=True)

# 2) right: L2‐error data (or stiffness—just match your filenames)
for fname in sorted(glob.glob('solver_*_l2_1e8.dat'),
                    key=lambda fn: int(re.search(r'solver_(\d+)_', fn).group(1))):
    if os.path.getsize(fname) == 0:
        continue
    m = re.search(r'solver_(\d+)_l2', fname)
    if not m:
        continue
    st = int(m.group(1))
    try:
        data = np.loadtxt(fname, comments='#')
    except ValueError:
        continue
    if data.ndim != 2 or data.shape[1] < 4:
        continue
    eps, dt, calls, err = data.T
    ax2.loglog(eps, calls, 's-', label=labels.get(st, str(st)))

ax2.set_xlabel(r'$\epsilon=|\mathbf{m}-\mathbf{m}_{\mathrm{ana}}|_2$')
ax2.set_title(r'$N$ vs $\epsilon$')
ax2.grid(True, which='both', ls='--', alpha=0.5)
ax2.legend(title='Solver', loc='best')
ax2.tick_params(direction='in', which='both', top=True, right=True)

# shared y‐label
ax1.set_ylabel('Number of derivative calls')

# finalize everything once, then save/show
fig.tight_layout()
fig.savefig('solver_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

