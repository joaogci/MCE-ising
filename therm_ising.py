import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os

SAVE_FOLDER = lambda L, lattice, Sz_vals: f"results/MCE_L{L}_{lattice}_npos{Sz_vals}/"
SAVE_FOLDER_FIGS = lambda L, lattice, Sz_vals: f"figures/MCE_L{L}_{lattice}_npos{Sz_vals}/"
SAVE_EXT_FIGS = ".pdf"
SAVE_EXT = ".dat"

NN_list = {"SS": 4, "SC": 6, "HCP": 12, "Hex": 8, "FCC": 12, "BCC": 8}
N_list = {"SS": lambda L: L**2, "SC": lambda L: L**3, "HCP": lambda L: 4*L**3, "Hex": lambda L: L**3, "FCC": lambda L: 4*L**3, "BCC": lambda L: 2*L**3}

# System information
L = 16
lattice = "SS"
Sz_vals = 2

N = N_list[lattice](L)
NN = NN_list[lattice]
S = (Sz_vals - 1.0) / 2.0

J = 1.0

H_plot = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

T_max = 5.0
T_vals = 64
H_max = 0.1
H_vals = 21

# Temperature, field, volume and J
T = np.arange(T_max / T_vals, T_max + T_max / T_vals, T_max / T_vals)
H = np.arange(0.0, H_max + H_max / (H_vals - 1), H_max / (H_vals - 1))

print("Computing thermodynamic variables for: ")
print(f" L: {L} | lattice: {lattice} | N: {N} | S: {S} | J: {J}")
print(f" Ti: {T[0]} | Tf: {T[-1]} | nT: {len(T)}")
print(f" Hi: {H[0]} | Hf: {H[-1]} | nH: {len(H)}")

# Energy and magnetization values and JDOS
max_E = 4.0 * S**2 * NN * N / 2.0
dE = 4
max_M = 2 * S * N
dM = 2

E_sys = np.arange(- max_E, max_E + dE, dE)
M_sys = np.arange(- max_M, max_M + dM, dM)
M_vals = len(M_sys)
E_vals = len(E_sys)

JDOS_filename = "JDOS/JDOS_L" + str(L) + "_" + lattice + "_npos" + str(Sz_vals) + ".dat"
print(f"Reading JDOS from file: {JDOS_filename}")

g = np.loadtxt(JDOS_filename)
if g[0, -1] == 0:
    g[:, (M_vals//2)+1:] = g[:, (M_vals//2)-1::-1]
print("JDOS read")

# Partition function
ln_ZM = np.zeros((H_vals, T_vals, M_vals))

print("Checking if ln_Z is already computed")
if os.path.isdir(SAVE_FOLDER(L, lattice, Sz_vals)):
    for q, m in enumerate(M_sys):
        ln_ZM[:, :, q] = np.loadtxt(SAVE_FOLDER(L, lattice, Sz_vals) + f"lnZ_q{q}" + SAVE_EXT)
    print("File found.")
else:
    print(f"File not found. Iterations for Z: {H_vals * M_vals * T_vals}")
    for q, m in enumerate(tqdm(M_sys)):
        hits = np.where(g[:, q] != 0.0)[0]
        energy = J * E_sys

        for j, h in enumerate(H):
            for i, t in enumerate(T):
                
                cte = np.log(g[hits[0], q]) - (energy[hits[0]] - h * m) / t
                ln_ZM[j, i, q] += cte
                ln_ZM[j, i, q] += np.log(1 + np.sum(np.exp(np.log(g[hits[1:], q]) - ((energy[hits[1:]] - h * m) / t) - cte)))
        
    print("Computed partition function")

# Minimization of G
G_tmp = np.zeros((H_vals, T_vals, M_vals))
G = np.zeros((H_vals, T_vals))
M = np.zeros((H_vals, T_vals))

for j, h in enumerate(H):
    for q, m in enumerate(M_sys):
        G_tmp[j, :, q] = - T * ln_ZM[j, :, q]

    G[j, :] = np.min(G_tmp[j, :, :], axis=1)
    M[j, :] = np.abs(M_sys[np.argmin(G_tmp[j, :, :], axis=1)])

G /= N
M /= N

S = - np.gradient(G, T, axis=1)
C = T * np.gradient(S, T, axis=1)
X = np.gradient(M, H, axis=0)
U = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    U[j, :] = G[j, :] + T * S[j, :]

print("Computed minimization of G and variables")

# MCE calculations
dSM = np.zeros((H_vals, T_vals))
# dT = np.zeros((H_vals, T_vals))
dM_dT_H = np.gradient(M, T, axis=1)

for i, t in enumerate(T):
    for j, h in enumerate(H):
        dSM[j, i] = np.trapz(dM_dT_H[:j+1, i], H[:j+1])
        # dT[j, i] = t * (np.exp(-np.trapz(np.interp(H_new, H[:j+1], dM_dT_H[:j+1, i] / C[:j+1, i]), H_new)) - 1)

print("MCE calculations done")

# Find Tc for H = 0
T_interp = np.linspace(T[0], T[-1], int(1e7))
grad = - np.gradient(np.interp(T_interp, T, M[0, :]), T_interp)
Tc = T_interp[np.where(np.max(grad) == grad)[0][0]]
print(f"Tc = {Tc}")

print("Tc calculations done")

# Saving results
if not os.path.isdir(SAVE_FOLDER(L, lattice, Sz_vals)):
    if not os.path.isdir("results/"):
        os.mkdir("results/")
        
    print(f"Saving Z on {SAVE_FOLDER(L, lattice, Sz_vals)}lnZ{SAVE_EXT}")
    os.mkdir(SAVE_FOLDER(L, lattice, Sz_vals))
    
    for q, m in enumerate(M_sys):
        np.savetxt(SAVE_FOLDER(L, lattice, Sz_vals) + f"lnZ_q{q}" + SAVE_EXT, ln_ZM[:, :, q])

# Global plotting options
plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams['lines.linewidth'] = '1'
matplotlib.rcParams['lines.markersize'] = '10'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['figure.subplot.left'] = '0.1'
matplotlib.rcParams['figure.subplot.bottom'] = '0.1'
matplotlib.rcParams['figure.subplot.right'] = '0.97'
matplotlib.rcParams['figure.subplot.top'] = '0.97'
matplotlib.rcParams['figure.subplot.wspace'] = '0.4'
matplotlib.rcParams['figure.subplot.hspace'] = '0.2'

if not os.path.isdir(SAVE_FOLDER_FIGS(L, lattice, Sz_vals)):
    if not os.path.isdir("figures/"):
        os.mkdir("figures/")
    os.mkdir(SAVE_FOLDER_FIGS(L, lattice, Sz_vals))

# Plots
plt.subplots(1, 3)
plt.figure(1)

plt.subplot(1, 3, 1)
skip = 2
colors = plt.cm.viridis(np.linspace(0, 1, T_vals // skip))
for i, t in enumerate(T):
    if i % skip == 0:
        plt.plot(H, M[:, i], color=colors[i//2])
plt.plot(H, M[:, 0], color=colors[0], label=f"T = {T[0]}")
plt.plot(H, M[:, -1], color=colors[-1], label=f"T = {T[-1]}")
plt.xlabel(r"$H$")
plt.ylabel(r"$M$")
plt.legend()

plt.subplot(1, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(H_plot) + 1))
i = 0
for j, h in enumerate(H):
    if h not in H_plot and h != 0.0:
        continue
    plt.plot(T, M[j, :], color=colors[i], label=f"H = {h}")
    i += 1
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")
plt.legend()

plt.subplot(1, 3, 3)
colors = plt.cm.viridis(np.linspace(0, 1, len(H_plot)))
i = 0
for j, h in enumerate(H):
    if h not in H_plot:
        continue
    plt.plot(T, -dSM[j, :], color=colors[i], label=f"H: {H[0]} to {h}")
    i += 1
plt.xlabel(r"$T$")
plt.ylabel(r"$-\Delta S_M$")
plt.legend()

plt.savefig(SAVE_FOLDER_FIGS(L, lattice, Sz_vals) + f"MCE_magnetization" + SAVE_EXT_FIGS)

plt.subplots(2, 3, sharex="all")
plt.figure(2)

plt.subplot(2, 3, 1)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, M[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")

plt.subplot(2, 3, 3)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, S[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$S$")

plt.subplot(2, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, U[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")

plt.subplot(2, 3, 6)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, C[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$C$")

plt.subplot(2, 3, 4)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, X[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$\chi$")

plt.subplot(2, 3, 5)
colors = plt.cm.viridis(np.linspace(0, 1, H_vals))
for j, h in enumerate(H):
    plt.plot(T, G[j, :], color=colors[j], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$G$")

plt.savefig(SAVE_FOLDER_FIGS(L, lattice, Sz_vals) + f"TD" + SAVE_EXT_FIGS)

print(f"Figures saved on {SAVE_FOLDER_FIGS(L, lattice, Sz_vals)}")

plt.show()
