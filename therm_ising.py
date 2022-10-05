import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['lines.marker'] = '.'
matplotlib.rcParams['lines.linestyle'] = '-'
matplotlib.rcParams['lines.linewidth'] = '1'
matplotlib.rcParams['lines.markersize'] = '5'
matplotlib.rcParams['savefig.bbox'] = 'tight'

SAVE_FOLDER = "figures/"
SAVE_EXT = ".pdf"

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

T_max = 5.0
dT = 0.05
H_max = 0.1
dH = 0.002

T = np.arange(dT, T_max + dT, dT)
H = np.arange(0.0, H_max + dH, dH)
# H = [0.0, 0.05, 0.1]
# dH = 0.05
T_vals = len(T)
H_vals = len(H)

print("Computing thermodynamic variables for: ")
print(f" L: {L} | lattice: {lattice} | N: {N} | S: {S} | J: {J}")
print(f" Ti: {T[0]} | Tf: {T[-1]} | nT: {len(T)}")
print(f" Hi: {H[0]} | Hf: {H[-1]} | nH: {len(H)}")

# Hamiltonian function
def Hamiltonian(J, E, h, M):
    return J * E - h * M

# Energy and magnetization values and JDOS
max_E = 4.0 * J * S**2 * NN * N / 2.0
dE = 4 * J
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

print(f"iterations for Z: {H_vals * M_vals * T_vals}")

for j, h in enumerate(tqdm(H)):
    for i, t in enumerate(T):
        for q, m in enumerate(M_sys):
            hits = np.where(g[:, q] != 0.0)[0]
            
            ln_ZM[j, i, q] += np.log(g[hits[0], q]) - Hamiltonian(J, E_sys[hits[0]], h, m) / t
            ln_ZM[j, i, q] += np.log(1 + np.sum(np.exp(np.log(g[hits[1:], q]) - (Hamiltonian(J, E_sys[hits[1:]], h, m) / t) - ln_ZM[j, i, q])))
        
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

plt.subplots(2, 3, sharex="all")
plt.figure(1)
plt.subplot(2, 3, 1)
for j, h in enumerate(H):
    plt.plot(T, M[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")

plt.subplot(2, 3, 3)
for j, h in enumerate(H):
    plt.plot(T, S[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$S$")

plt.subplot(2, 3, 2)
for j, h in enumerate(H):
    plt.plot(T, U[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")

plt.subplot(2, 3, 6)
for j, h in enumerate(H):
    plt.plot(T, C[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$C$")

plt.subplot(2, 3, 4)
for j, h in enumerate(H):
    plt.plot(T, X[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$\chi$")

plt.subplot(2, 3, 5)
for j, h in enumerate(H):
    plt.plot(T, G[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$G$")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig(SAVE_FOLDER + f"TD_Ising_L{L}_{lattice}_npos{Sz_vals}" + SAVE_EXT)

plt.subplots(1, 2, sharex="all")
plt.figure(2)

plt.subplot(1, 2, 1)
for j, h in enumerate(H):
    plt.plot(T, -dSM[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$- \Delta S_M$ from MR")
# plt.legend()

plt.subplot(1, 2, 2)
for j, h in enumerate(H):
    plt.plot(T, -dSM_2[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$- \Delta S_M$ from $\Delta S$")
# plt.legend()

# plt.subplot(1, 3, 3)
# for j, h in enumerate(H):
#     plt.plot(T, dT[j, :], label=h)
# plt.xlabel(r"$T$")
# plt.ylabel(r"$\Delta T$")
# # plt.ylim([-0.05, 0.43])
# # plt.legend()

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig(SAVE_FOLDER + f"MCE_Ising_L{L}_{lattice}_npos{Sz_vals}" + SAVE_EXT)

plt.show()
