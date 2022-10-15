import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Global plotting options
plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams['lines.linewidth'] = '1'
matplotlib.rcParams['lines.markersize'] = '7.5'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['figure.subplot.left'] = '0.1'
matplotlib.rcParams['figure.subplot.bottom'] = '0.1'
matplotlib.rcParams['figure.subplot.right'] = '0.97'
matplotlib.rcParams['figure.subplot.top'] = '0.97'
matplotlib.rcParams['figure.subplot.wspace'] = '0.2'
matplotlib.rcParams['figure.subplot.hspace'] = '0.2'
matplotlib.rcParams['figure.figsize'] = (5, 4)

# Plots
Jp = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

plt.figure(1)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(Jp) + 1))

min_T_idx = 30
max_T_idx = 55

dSM_0 = np.loadtxt("results/dSM_MCE_L8_SC_npos2.dat")
dSM_0_max = np.max(-dSM_0[:, 0])
Tc = dSM_0[np.where(dSM_0_max == - dSM_0[:, 0])[0][0], 1]

plt.plot(dSM_0[min_T_idx:max_T_idx, 1] / Tc, -dSM_0[min_T_idx:max_T_idx, 0] / dSM_0_max, color=colors[0], label="Jp = 0.0")

for i, J in enumerate(Jp):
    dSM = np.loadtxt(f"results/dSM_GMCE_L8_SC_npos2_Jp{J}_K50.0.dat")
    plt.plot(dSM[min_T_idx:max_T_idx, 1] / Tc, -dSM[min_T_idx:max_T_idx, 0] / dSM_0_max, color=colors[i+1], label=f"Jp = {J}")

plt.xlabel(r"$T / T_C^{J^\prime=0}$")
plt.ylabel(r"$-\Delta S_M / \Delta S_M^{max; J^\prime=0}$")
plt.legend()

plt.savefig("figures/dSM_comparison_Jp_npos2.eps")

Jp = [0.5, 1.0, 1.5, 2.0]

plt.figure(2)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(Jp) + 1))

min_T_idx = 95
max_T_idx = 145

dSM_0 = np.loadtxt("results/dSM_MCE_L8_SC_npos3.dat")
dSM_0_max = np.max(-dSM_0[:, 0])
Tc = dSM_0[np.where(dSM_0_max == - dSM_0[:, 0])[0][0], 1]

plt.plot(dSM_0[min_T_idx:max_T_idx, 1] / Tc, -dSM_0[min_T_idx:max_T_idx, 0] / dSM_0_max, color=colors[0], label="Jp = 0.0")

for i, J in enumerate(Jp):
    dSM = np.loadtxt(f"results/dSM_GMCE_L8_SC_npos3_Jp{J}_K50.0.dat")
    plt.plot(dSM[min_T_idx:max_T_idx, 1] / Tc, -dSM[min_T_idx:max_T_idx, 0] / dSM_0_max, color=colors[i+1], label=f"Jp = {J}")

plt.xlabel(r"$T / T_C^{J^\prime=0}$")
plt.ylabel(r"$-\Delta S_M / \Delta S_M^{max; J^\prime=0}$")
plt.legend()
plt.savefig("figures/dSM_comparison_Jp_npos3.eps")

plt.show()
