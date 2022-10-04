import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

NN_list = {"SS": 4, "SC": 6, "HCP": 12, "Hex": 8, "FCC": 12, "BCC": 8}
N_list = {"SS": lambda L: L**2, "SC": lambda L: L**3, "HCP": lambda L: 4*L**3, "Hex": lambda L: L**3, "FCC": lambda L: 4*L**3, "BCC": lambda L: 2*L**3}

# System information
L = 8
lattice = "SC"
Sz_vals = 2

N = N_list[lattice](L)
NN = NN_list[lattice]
S = (Sz_vals - 1.0) / 2.0

J = 1.0

T = np.linspace(0.01, 10.0, 50)
H = np.linspace(0.0, 1.0, 6)
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
for j, h in enumerate(H):
    for i, t in enumerate(T):
        for q, m in enumerate(M_sys):
            hits = np.where(g[:, q] != 0.0)[0]
            
            ln_ZM[j, i, q] += np.log(g[hits[0], q]) - Hamiltonian(J, E_sys[hits[0]], h, m) / t
            ln_ZM[j, i, q] += np.log(1 + np.sum(np.exp(np.log(g[hits[1:], q]) - (Hamiltonian(J, E_sys[hits[1:]], h, m) / t) - ln_ZM[j, i, q])))
        
print("Computed partition function")

# Minimization of G
G_tmp = np.zeros((H_vals, T_vals, M_vals))
G = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    for q, m in enumerate(M_sys):
        G_tmp[j, :, q] = - T * ln_ZM[j, :, q]

        G[j, :] = np.min(G_tmp[j, :, :], axis=1)

G /= N

M = - np.gradient(G, H, axis=0)
S = - np.gradient(G, T, axis=1)
C = - T * np.gradient(np.gradient(G, T, axis=1), T, axis=1)
X = - np.gradient(np.gradient(G, H, axis=0), H, axis=0)
U = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    U[j, :] = G[j, :] + T * S[j, :] + H[j] * M[j, :]

print("Computed minimization of G and variables")

plt.figure(1)
for j, h in enumerate(H):
    plt.plot(T, M[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$M$")
# plt.legend()

plt.figure(2)
for j, h in enumerate(H):
    plt.plot(T, S[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$S$")
# plt.legend()

plt.figure(3)
for j, h in enumerate(H):
    plt.plot(T, U[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
# plt.legend()

plt.figure(4)
for j, h in enumerate(H):
    plt.plot(T, C[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$C$")
# plt.legend()

plt.figure(5)
for j, h in enumerate(H):
    plt.plot(T, X[j, :], label=h)
plt.xlabel(r"$T$")
plt.ylabel(r"$\chi$")
# plt.legend()

plt.show()
