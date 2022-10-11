import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

SAVE_FOLDER = "figures/"
SAVE_EXT = ".pdf"

NN_list = {"SS": 4, "SC": 6, "HCP": 12, "Hex": 8, "FCC": 12, "BCC": 8}
N_list = {"SS": lambda L: L**2, "SC": lambda L: L**3, "HCP": lambda L: 4*L**3, "Hex": lambda L: L**3, "FCC": lambda L: 4*L**3, "BCC": lambda L: 2*L**3}

# System information
L = 8
lattice = "SC"
Sz_vals = 2

N = N_list[lattice](L)
NN = NN_list[lattice]
S = (Sz_vals - 1.0) / 2.0

Jp = 3.5
K = 50.0

T_max = 14.0
dT = T_max / 64 # 0.05
H_max = 0.1
dH = H_max / 20 # 0.002
v_max = (0.5 * NN * Jp) / K
dv = v_max / (51 - 2) # 0.01

T = np.arange(dT, T_max + dT, dT)
H = np.arange(0.0, H_max + dH, dH)
v = np.arange(-dv, v_max + dv, dv)
T_vals = len(T)
H_vals = len(H)
v_vals = len(v)

print("Computing thermodynamic variables for: ")
print(f" L: {L} | lattice: {lattice} | N: {N} | S: {S} ")
print(f" K: {K} | Jp: {Jp} ")
print(f" Ti: {T[0]} | Tf: {T[-1]} | nT: {len(T)}")
print(f" Hi: {H[0]} | Hf: {H[-1]} | nH: {len(H)}")    
print(f" vi: {v[0]} | vf: {v[-1]} | nv: {len(v)}")    

pre_comp = 0.5 * K * N * np.power(v, 2.0)
# Hamiltonian function
def Hamiltonian(k, E, h, M):
    return J[k] * E - h * M + pre_comp[k]

# Energy and magnetization values and JDOS
max_E = 4.0 * S**2 * NN * N / 2.0
dE = 4
max_M = 2 * S * N
dM = 2

E_sys = np.arange(- max_E, max_E + dE, dE)
M_sys = np.arange(- max_M, max_M + dM, dM)
J = 1 + Jp * v
M_vals = len(M_sys)
E_vals = len(E_sys)

JDOS_filename = "JDOS/JDOS_L" + str(L) + "_" + lattice + "_npos" + str(Sz_vals) + ".dat"
print(f"Reading JDOS from file: {JDOS_filename}")

g = np.loadtxt(JDOS_filename)
if g[0, -1] == 0:
    g[:, (M_vals//2)+1:] = g[:, (M_vals//2)-1::-1]
print("JDOS read")

# Partition function
ln_ZMv = np.zeros((H_vals, T_vals, M_vals, v_vals))

print(f"iterations for Z: {H_vals * v_vals * M_vals * T_vals}")

for j, h in enumerate(tqdm(H)):
    for i, t in enumerate(T):
        for q, m in enumerate(M_sys):
            hits = np.where(g[:, q] != 0.0)[0]
            
            for k in range(v_vals):
                cte = np.log(g[hits[0], q]) - Hamiltonian(k, E_sys[hits[0]], h, m) / t
                ln_ZMv[j, i, q, k] += cte
                ln_ZMv[j, i, q, k] += np.log(1 + np.sum(np.exp(np.log(g[hits[1:], q]) - (Hamiltonian(k, E_sys[hits[1:]], h, m) / t) - cte)))

print("Computed partition function")

# Minimization of G
G_tmp = np.zeros((H_vals, T_vals, M_vals, v_vals))
G_min_vol = np.zeros((H_vals, T_vals, M_vals))
indx_min_vol = np.zeros((H_vals, T_vals, M_vals), dtype=int)
idx_min_mag = np.zeros((H_vals, T_vals), dtype=int)

G = np.zeros((H_vals, T_vals))
M = np.zeros((H_vals, T_vals))
v_min = np.zeros((H_vals, T_vals))

for j, h in enumerate(H):
    for q, m in enumerate(M_sys):
        for k in range(v_vals):
            G_tmp[j, :, q, k] = - T * ln_ZMv[j, :, q, k]

    # T, H, M
    G_min_vol[j, :, :] = np.min(G_tmp[j, :, :, :], axis=2)
    indx_min_vol[j, :, :] = np.argmin(G_tmp[j, :, :, :], axis=2)
    
    # T, H
    G[j, :] = np.min(G_min_vol[j, :, :], axis=1)
    idx_min_mag[j, :] = np.argmin(G_min_vol[j, :, :], axis=1)
    
    M[j, :] = np.abs(M_sys[idx_min_mag[j, :]])
    for i in range(T_vals):
        v_min[j, i] = v[indx_min_vol[j, i, idx_min_mag[j, i]]]
    
G /= N
M /= N

S = - np.gradient(G, T, axis=1)
C = T * np.gradient(S, T, axis=1)
if H_vals > 1:
    X = np.gradient(M, H, axis=0)
U = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    U[j, :] = G[j, :] + T * S[j, :]

print("Computed minimization of G and variables")

dSM = np.zeros((H_vals, T_vals))
dSM_2 = np.zeros((H_vals, T_vals))
# dT = np.zeros((H_vals, T_vals))
dM_dT_H = np.gradient(M, T, axis=1)

for i, t in enumerate(T):
    for j, h in enumerate(H):
        H_new = np.arange(H[0], H[j], dH / 1000)

        dSM[j, i] = np.trapz(np.interp(H_new, H[:j+1], dM_dT_H[:j+1, i]), H_new)
        dSM_2[j, i] = S[j, i] - S[0, i]
        # dT[j, i] = t * (np.exp(-np.trapz(np.interp(H_new, H[:j+1], dM_dT_H[:j+1, i] / C[:j+1, i]), H_new)) - 1)

print("MCE calculations done")


# Global plotting options
plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams['lines.linewidth'] = '1'
matplotlib.rcParams['lines.markersize'] = '5'
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['figure.subplot.left'] = '0.1'
matplotlib.rcParams['figure.subplot.bottom'] = '0.1'
matplotlib.rcParams['figure.subplot.right'] = '0.9'
matplotlib.rcParams['figure.subplot.top'] = '0.9'
matplotlib.rcParams['figure.subplot.wspace'] = '0.4'
matplotlib.rcParams['figure.subplot.hspace'] = '0.4'

# Plots
plt.subplots(1, 3)
plt.figure(1)

plt.subplot(1, 3, 1)
colors = plt.cm.viridis(np.linspace(0,1,T_vals//1))
for i, t in enumerate(T):
    if i % 1 == 0:
        plt.plot(H, M[:, i], color=colors[i])
plt.plot(H, M[:, 0], color=colors[0], label=f"T = {T[0]}")
plt.plot(H, M[:, -1], color=colors[-1], label=f"T = {T[-1]}")
plt.xlabel(r"$H$")
plt.ylabel(r"$M$")
plt.legend()

plt.subplot(1, 3, 2)
colors = plt.cm.viridis(np.linspace(0,1,H_vals))
i = 0
for j, h in enumerate(H):
    # if h not in H_plot and h != 0.0:
    #     continue
    plt.plot(T, M[j, :], color=colors[i], label=f"H = {h}")
    i += 1
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")
plt.legend()

plt.subplot(1, 3, 3)
colors = plt.cm.viridis(np.linspace(0,1,H_vals))
i = 0
for j, h in enumerate(H):
    # if h not in H_plot:
    #     continue:w
    
    plt.plot(T, -dSM[j, :], color=colors[i], label=f"H: {H[0]} to {h}")
    i += 1
plt.xlabel(r"$T$")
plt.ylabel(r"$-\Delta S_M$")
plt.legend()

plt.figure(2)
plt.plot(T, v_min[0, :])
plt.xlabel(r"$T$")
plt.ylabel(r"$v$")


# plt.subplots(2, 3, sharex="all")
# plt.figure(1)
# plt.subplot(2, 3, 1)
# for j, h in enumerate(H):
#     plt.plot(T, M[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$M$")

# plt.subplot(2, 3, 3)
# for j, h in enumerate(H):
#     plt.plot(T, S[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$S$")

# plt.subplot(2, 3, 2)
# for j, h in enumerate(H):
#     plt.plot(T, U[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$U$")

# plt.subplot(2, 3, 6)
# for j, h in enumerate(H):
#     plt.plot(T, C[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$C$")

# plt.subplot(2, 3, 4)
# for j, h in enumerate(H):
#     plt.plot(T, X[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$\chi$")

# plt.subplot(2, 3, 5)
# for j, h in enumerate(H):
#     plt.plot(T, G[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$G$")

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
# # plt.savefig(SAVE_FOLDER + f"TD_Ising_L{L}_{lattice}_npos{Sz_vals}" + SAVE_EXT)


# plt.subplots(1, 2, sharex="all")
# plt.figure(2)

# plt.subplot(1, 2, 1)
# for j, h in enumerate(H):
#     plt.plot(T, -dSM[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$- \Delta S_M$ from MR")
# # plt.legend()

# plt.subplot(1, 2, 2)
# for j, h in enumerate(H):
#     plt.plot(T, -dSM_2[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$- \Delta S_M$ from $\Delta S$")
# # plt.legend()

# # plt.subplot(1, 3, 3)
# # for j, h in enumerate(H):
# #     plt.plot(T, dT[j, :], label=h)
# # plt.xlabel(r"$T$")
# # plt.ylabel(r"$\Delta T$")
# # # plt.ylim([-0.05, 0.43])
# # # plt.legend()

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
#                     hspace=0.4)
# # plt.savefig(SAVE_FOLDER + f"MCE_Ising_L{L}_{lattice}_npos{Sz_vals}" + SAVE_EXT)

plt.show()
