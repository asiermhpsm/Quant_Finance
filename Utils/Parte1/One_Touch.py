import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# Parámetros
r = 0.05          # tasa libre de riesgo
sigma = 0.2       # volatilidad
T = 1             # tiempo a vencimiento
Su = 1.5          # barrera superior (call)
Sl = 0.5          # barrera inferior (put)

# Mallas para S (precio del subyacente) y t (tiempo)
S = np.linspace(0.01, 2, 100)    # evita dividir por 0
t = np.linspace(0, T-0.001, 100) # evita T-t = 0
SS, TT = np.meshgrid(S, t)
tau = T - TT  # tiempo restante hasta vencimiento

# --- One-touch Call ---
d1_call = (np.log(SS / Su) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
d6_call = (np.log(SS / Su) - (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

V_call = (Su / SS)**(2 * r / sigma**2) * norm.cdf(d6_call) + (SS / Su) * norm.cdf(d1_call)

# Gráfica 3D del valor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(SS, TT, V_call, cmap='viridis', edgecolor='none')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('Valor')
ax.view_init(30, 225)
plt.grid(True)
plt.savefig(r'Imagenes/Parte1/9_Americanas/One_Touch_Call.pdf', format="pdf", bbox_inches="tight")
plt.close()

# --- Delta numérica (∂V/∂S) ---
dS = S[1] - S[0]  # paso en S
delta_call, _ = np.gradient(V_call, dS, t[1] - t[0])  # solo nos interesa la derivada en S

# Gráfica 3D de la Delta
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(SS, TT, delta_call, cmap='viridis', edgecolor='none')
ax.set_xlabel('PS')
ax.set_ylabel('Tiempo t')
ax.set_zlabel('Delta')
ax.view_init(30, 225)
plt.grid(True)
plt.savefig(r'Imagenes/Parte1/9_Americanas/Delta_One_Touch_Call.pdf', format="pdf", bbox_inches="tight")
plt.close()

