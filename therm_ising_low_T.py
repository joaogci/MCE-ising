import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# System information
L = 8
N = L**2
lattice = "SS"
NN = 4 if lattice == "SS" else 6
log_REP = 3

J = 1.0

T = np.linspace(0.01, 5.0, 32)
H = np.linspace(0.0, 1.0, 32)
T_vals = len(T)
H_vals = len(H)

print("Computing thermodynamic variables for: ")
print(f" L: {L} | lattice: {lattice} | J: {J}")
print(f" Ti: {T[0]} | Tf: {T[-1]} | nT: {len(T)}")
print(f" Hi: {H[0]} | Hf: {H[-1]} | nH: {len(H)}")

# Hamiltonian function
def Hamiltonian(J, E, h, M):
    return J * E - h * M

# Energy and magnetization values and JDOS
E = np.arange(- J * NN * N / 2, J * NN * N / 2 + 4 * J, 4 * J)
M = np.arange(- N, N + 1, 2)
M_vals = len(M)
E_vals = len(E)

JDOS_filename = "JDOS/JDOS_L" + str(L) + "_" + lattice + "_npos2_R1E" + str(log_REP) + ".txt"
print(f"Reading JDOS from file: {JDOS_filename}")

g = np.loadtxt(JDOS_filename)
if g[0, -1] == 0:
    g[:, (M_vals//2)+1:] = g[:, (M_vals//2)-1::-1]
print("JDOS read")

# Partition function
# Z = np.zeros((H_vals, T_vals))
ln_ZM = np.zeros((H_vals, T_vals, M_vals))

for j, h in enumerate(H):
    for i, t in enumerate(T):
        for q, m in enumerate(M):
            hits = np.where(g[:, q] != 0.0)[0]
            
            ln_ZM[j, i, q] += np.log(g[hits[0], q]) - Hamiltonian(J, E[hits[0]], h, m) / t
            ln_ZM[j, i, q] += np.log(1 + np.sum(np.exp(np.log(g[hits[1:], q]) - (Hamiltonian(J, E[hits[1:]], h, m) / t) - ln_ZM[j, i, q])))
        
        # Z[j, i] += np.sum(ZM[j, i, :])

print("Computed partition function")

# Ensemble averages
# Energies
# E_mean = np.zeros((H_vals, T_vals))
# E2_mean = np.zeros((H_vals, T_vals))
# C_mean = np.zeros((H_vals, T_vals))

# for j, h in enumerate(H):
#     for i, t in enumerate(T):
#         for idx, e in enumerate(E):
#             E_mean[j, i] += e * np.sum(g[idx, :] * np.exp(- Hamiltonian(J, e, h, M) / t)) / Z[j, i]
#             E2_mean[j, i] += e**2 * np.sum(g[idx, :] * np.exp(- Hamiltonian(J, e, h, M) / t)) / Z[j, i]
#         C_mean[j, i] = (E2_mean[j, i] - E_mean[j, i]**2) / t**2

# E_mean /= N
# E2_mean /= N
# C_mean /= N

# # Magnetizations
# M_mean = np.zeros((H_vals, T_vals))
# M2_mean = np.zeros((H_vals, T_vals))
# M4_mean = np.zeros((H_vals, T_vals))
# Mabs_mean = np.zeros((H_vals, T_vals))
# X_mean = np.zeros((H_vals, T_vals))

# for j, h in enumerate(H):
#     for i, t in enumerate(T):
#         for q, m in enumerate(M):
#             M_mean[j, i] += m * ZM[j, i, q] / Z[j, i]
#             M2_mean[j, i] += m**2 * ZM[j, i, q] / Z[j, i]
#             M4_mean[j, i] += m**4 * ZM[j, i, q] / Z[j, i]
#             Mabs_mean[j, i] += np.abs(m) * ZM[j, i, q] / Z[j, i]
#         X_mean[j, i] = (M2_mean[j, i] - M_mean[j, i]**2) / t

# M_mean /= N
# M2_mean /= N
# M4_mean /= N
# Mabs_mean /= N
# X_mean /= N

# print("Computed ensemble average variables")

# Minimization of G
G_tmp = np.zeros((H_vals, T_vals, M_vals))
G = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    for q, m in enumerate(M):
        G_tmp[j, :, q] = - T * ln_ZM[j, :, q]

        G[j, :] = np.min(G_tmp[j, :, :], axis=1)

G /= N

M_F = - np.gradient(G, H, axis=0)
S_F = - np.gradient(G, T, axis=1)
C_F = - T * np.gradient(np.gradient(G, T, axis=1), T, axis=1)
X_F = - np.gradient(np.gradient(G, H, axis=0), H, axis=0)
U_F = np.zeros((H_vals, T_vals))
for j, h in enumerate(H):
    U_F[j, :] = G[j, :] + T * S_F[j, :] + H[j] * M_F[j, :]

print("Computed minimization of G and variables")

plt.figure(1)
plt.plot(T, M_F[0, :], label="from F min")
# plt.plot(T, Mabs_mean[0, :], label="from T avg")
plt.xlabel(r"$T$")
plt.ylabel(r"$|m|$")
plt.legend()

plt.figure(2)
plt.plot(T, S_F[0, :], label="from F min")
plt.xlabel(r"$T$")
plt.ylabel(r"$S$")
plt.legend()

plt.figure(3)
plt.plot(T, U_F[0, :], label="from F min")
# plt.plot(T, E_mean[0, :], label="from T avg")
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.legend()

plt.figure(4)
plt.plot(T, C_F[0, :], label="from F min")
# plt.plot(T, C_mean[0, :], label="from T avg")
plt.xlabel(r"$T$")
plt.ylabel(r"$C$")
plt.legend()

plt.figure(5)
plt.plot(T, X_F[0, :], label="from F min")
# plt.plot(T, X_mean[0, :], label="from T avg")
plt.xlabel(r"$T$")
plt.ylabel(r"$\chi$")
plt.legend()

plt.show()
