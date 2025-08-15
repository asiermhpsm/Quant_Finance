import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid 

# Parámetros
mu = 0.15
sigma_real = 0.5
sigma_impl = 0.3
r = 0.15
D = 0.01
S0 = 100
E = 100
T = 1
N = 100
M = 10

dt = T / N
t = np.linspace(0, T, N + 1)
S_paths = np.zeros((M, N + 1))
S_paths[:, 0] = S0

# Simular trayectorias de S (GBM)
for i in range(M):
    W = np.cumsum(np.sqrt(dt) * np.random.randn(N))
    S_paths[i, 1:] = S0 * np.exp((mu - 0.5 * sigma_real**2) * t[1:] + sigma_real * W)

# Prealocar variables
Gamma_i = np.zeros((M, N + 1))
dPi = np.zeros((M, N + 1))
Pi = np.zeros((M, N + 1))

# Cálculo de Gamma^i y dPi
for i in range(M):
    for j in range(N):
        tau = T - t[j]
        if tau > 0:
            S = S_paths[i, j]
            d1 = (np.log(S / E) + (r - D + 0.5 * sigma_impl**2) * tau) / (sigma_impl * np.sqrt(tau))
            N_prime = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
            Gamma_i[i, j] = np.exp(-D * tau) * N_prime / (sigma_impl * S * np.sqrt(tau))
            dPi[i, j] = 0.5 * (sigma_real**2 - sigma_impl**2) * S**2 * Gamma_i[i, j] * dt

    # Integral acumulada 
    Pi[i, :] = cumulative_trapezoid(dPi[i, :] * np.exp(r * (t - t[0])), t, initial=0)

# Graficar trayectorias de Pi
plt.figure()
plt.plot(t, Pi.T)
plt.xlabel('Tiempo')
plt.ylabel('Beneficio')
plt.grid(True)


plt.savefig(r'Imagenes/Parte1/10_Cobertura/Cobertura_Volat_Imp.pdf', format="pdf", bbox_inches="tight")
plt.close()
