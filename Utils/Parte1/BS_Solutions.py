import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# --- Parámetros generales ---
E = 20        # Precio de ejercicio
r = 0.05       # Tasa libre de riesgo
D = 0.0       # Tasa de dividendos
sigma = 0.2    # Volatilidad
T = 1          # Tiempo hasta vencimiento

# --- Funciones auxiliares d1, d2 ---
def d1(S, tau):
    return (np.log(S / E) + (r - D + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def d2(S, tau):
    return d1(S, tau) - sigma * np.sqrt(tau)

Np = norm.pdf
Nc = norm.cdf

# --- Call Europea ---
def value_call(S, t, tau):
    return S * np.exp(-D * tau) * Nc(d1(S, tau)) - E * np.exp(-r * tau) * Nc(d2(S, tau))

def delta_call(S, t, tau):
    return np.exp(-D * tau) * Nc(d1(S, tau))

def gamma_call(S, t, tau):
    return np.exp(-D * tau) * Np(d1(S, tau)) / (sigma * S * np.sqrt(tau))

def theta_call(S, t, tau):
    return (-sigma * S * np.exp(-D * tau) * Np(d1(S, tau)) / (2 * np.sqrt(tau))
            + D * S * np.exp(-D * tau) * Nc(d1(S, tau))
            - r * E * np.exp(-r * tau) * Nc(d2(S, tau)))

def speed_call(S, t, tau):
    return (-np.exp(-D * tau) * Np(d1(S, tau)) * (d1(S, tau) + sigma * np.sqrt(tau))
            / (sigma ** 2 * S ** 2 * tau))

def vega_call(S, t, tau):
    return S * np.sqrt(tau) * np.exp(-D * tau) * Np(d1(S, tau))

def rho_r_call(S, t, tau):
    return E * tau * np.exp(-r * tau) * Nc(d2(S, tau))

def rho_D_call(S, t, tau):
    return -S * tau * np.exp(-D * tau) * Nc(d1(S, tau))

# --- Put Europea ---
def value_put(S, t, tau):
    return -S * np.exp(-D * tau) * Nc(-d1(S, tau)) + E * np.exp(-r * tau) * Nc(-d2(S, tau))

def delta_put(S, t, tau):
    return np.exp(-D * tau) * (Nc(d1(S, tau)) - 1)

gamma_put = gamma_call  # igual que Call

def theta_put(S, t, tau):
    return (-sigma * S * np.exp(-D * tau) * Np(-d1(S, tau)) / (2 * np.sqrt(tau))
            - D * S * np.exp(-D * tau) * Nc(-d1(S, tau))
            + r * E * np.exp(-r * tau) * Nc(-d2(S, tau)))

speed_put = speed_call  # igual que Call
vega_put = vega_call    # igual que Call

def rho_r_put(S, t, tau):
    return -E * tau * np.exp(-r * tau) * Nc(-d2(S, tau))

def rho_D_put(S, t, tau):
    return S * tau * np.exp(-D * tau) * Nc(-d1(S, tau))

# --- Binary Call ---
def value_bin_call(S, t, tau):
    return np.exp(-r * tau) * Nc(d2(S, tau))

def delta_bin_call(S, t, tau):
    return np.exp(-r * tau) * Np(d2(S, tau)) / (sigma * S * np.sqrt(tau))

def gamma_bin_call(S, t, tau):
    return -np.exp(-r * tau) * d1(S, tau) * Np(d2(S, tau)) / (sigma ** 2 * S * tau)

def theta_bin_call(S, t, tau):
    return (r * np.exp(-r * tau) * Nc(d2(S, tau)) *
            (d1(S, tau) / (2 * tau) - (r - D) / (sigma * np.sqrt(tau))))

def speed_bin_call(S, t, tau):
    return (np.exp(-r * tau) * Np(d2(S, tau)) *
            (-2 * d1(S, tau) + (1 - d1(S, tau) * d2(S, tau)) / (sigma * np.sqrt(tau)))
            / (sigma ** 2 * S * tau))

def vega_bin_call(S, t, tau):
    return -np.exp(-r * tau) * Np(d2(S, tau)) * d1(S, tau) / (sigma * np.sqrt(tau))

def rho_r_bin_call(S, t, tau):
    return -tau * np.exp(-r * tau) * Nc(d2(S, tau)) + np.exp(-r * tau) * Np(d2(S, tau)) / sigma

def rho_D_bin_call(S, t, tau):
    return (np.sqrt(tau) / sigma) * np.exp(-r * tau) * Np(d2(S, tau))

# --- Binary Put ---
def value_bin_put(S, t, tau):
    return np.exp(-r * tau) * (1 - Nc(d2(S, tau)))

def delta_bin_put(S, t, tau):
    return -np.exp(-r * tau) * Np(d2(S, tau)) / (sigma * S * np.sqrt(tau))

def gamma_bin_put(S, t, tau):
    return np.exp(-r * tau) * d1(S, tau) * Np(d2(S, tau)) / (sigma ** 2 * S * tau)

def theta_bin_put(S, t, tau):
    return (r * np.exp(-r * tau) * (1 - Nc(d2(S, tau))) *
            (d1(S, tau) / (2 * tau) - (r - D) / (sigma * np.sqrt(tau))))

speed_bin_put = speed_bin_call  # misma expresión

def vega_bin_put(S, t, tau):
    return np.exp(-r * tau) * Np(d2(S, tau)) * d1(S, tau) / (sigma * np.sqrt(tau))

def rho_r_bin_put(S, t, tau):
    return -tau * np.exp(-r * tau) * (1 - Nc(d2(S, tau))) - np.exp(-r * tau) * Np(d2(S, tau)) / sigma

def rho_D_bin_put(S, t, tau):
    return - (np.sqrt(tau) / sigma) * np.exp(-r * tau) * Np(d2(S, tau))

# --- Gráficas ---
def plot_surf(T, funcion_evaluada, etiqueta_z, output_file=None):
    S = np.linspace(1, 200, 100)
    t = np.linspace(0, T - 0.001, 100)
    S_grid, t_grid = np.meshgrid(S, t)
    tau = T - t_grid

    Z = funcion_evaluada(S_grid, t_grid, tau)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, t_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('S')
    ax.set_ylabel('t')
    ax.set_zlabel(etiqueta_z)
    ax.view_init(elev=30, azim=-135)
    
    plt.grid(True)

    if output_file:
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        plt.close()


def plot_2D(T, funcion_evaluada, output_file_S=None, output_file_t=None):
    # Gráfico 2D con S fijado (S = 100)
    S_fijo = 100
    t_vec = np.linspace(0, T - 0.001, 500)
    tau_vec = T - t_vec
    Z_t = funcion_evaluada(S_fijo * np.ones_like(t_vec), t_vec, tau_vec)

    plt.figure()
    plt.plot(t_vec, Z_t, linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Valor')
    plt.title(f'S = {S_fijo}')
    plt.grid(True)
    if output_file_S:
        plt.savefig(output_file_S, format="pdf", bbox_inches="tight")
        plt.close()

    # Gráfico 2D con t fijado (t = 0.5)
    t_fijo = 0.5
    S_vec = np.linspace(1, 200, 500)
    tau_fijo = T - t_fijo
    Z_S = funcion_evaluada(S_vec, t_fijo * np.ones_like(S_vec), tau_fijo * np.ones_like(S_vec))

    plt.figure()
    plt.plot(S_vec, Z_S, linewidth=2)
    plt.xlabel('S')
    plt.ylabel('Valor')
    plt.title(f't = {t_fijo}')
    plt.grid(True)
    if output_file_t:
        plt.savefig(output_file_t, format="pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_surf(T, value_call, "Value", r'Imagenes/Parte1/6_Sols/Call/Call3D.pdf')
    plot_2D(T, value_call, r'Imagenes/Parte1/6_Sols/Call/CallSFijo.pdf', r'Imagenes/Parte1/6_Sols/Call/CalltFIjo.pdf')
    plot_surf(T, delta_call, "Delta", r'Imagenes/Parte1/6_Sols/Call/Call_Delta.pdf')
    plot_surf(T, gamma_call, "Gamma", r'Imagenes/Parte1/6_Sols/Call/Call_Gamma.pdf')
    plot_surf(T, theta_call, "Theta", r'Imagenes/Parte1/6_Sols/Call/Call_Theta.pdf')
    plot_surf(T, speed_call, "Speed", r'Imagenes/Parte1/6_Sols/Call/Call_Speed.pdf')
    plot_surf(T, vega_call, "Vega", r'Imagenes/Parte1/6_Sols/Call/Call_Vega.pdf')
    plot_surf(T, rho_r_call, "Rho (r)", r'Imagenes/Parte1/6_Sols/Call/Call_Rho_r.pdf')
    plot_surf(T, rho_D_call, "Rho (D)", r'Imagenes/Parte1/6_Sols/Call/Call_Rho_D.pdf')
    
    plot_surf(T, value_put, "Value", r'Imagenes/Parte1/6_Sols/Put/Put3D.pdf')
    plot_2D(T, value_put, r'Imagenes/Parte1/6_Sols/Put/PutSFijo.pdf', r'Imagenes/Parte1/6_Sols/Put/PuttFIjo.pdf')
    plot_surf(T, delta_put, "Delta", r'Imagenes/Parte1/6_Sols/Put/Put_Delta.pdf')
    plot_surf(T, gamma_put, "Gamma", r'Imagenes/Parte1/6_Sols/Put/Put_Gamma.pdf')
    plot_surf(T, theta_put, "Theta", r'Imagenes/Parte1/6_Sols/Put/Put_Theta.pdf')
    plot_surf(T, speed_put, "Speed", r'Imagenes/Parte1/6_Sols/Put/Put_Speed.pdf')
    plot_surf(T, vega_put, "Vega", r'Imagenes/Parte1/6_Sols/Put/Put_Vega.pdf')
    plot_surf(T, rho_r_put, "Rho (r)", r'Imagenes/Parte1/6_Sols/Put/Put_Rho_r.pdf')
    plot_surf(T, rho_D_put, "Rho (D)", r'Imagenes/Parte1/6_Sols/Put/Put_Rho_D.pdf')
    
    plot_surf(T, value_bin_call, "Value", r'Imagenes/Parte1/6_Sols/Binary_Call/BinaryCall3D.pdf')
    plot_2D(T, value_bin_call, r'Imagenes/Parte1/6_Sols/Binary_Call/BinaryCallSFijo.pdf', r'Imagenes/Parte1/6_Sols/Binary_Call/BinaryCalltFIjo.pdf')
    plot_surf(T, delta_bin_call, "Delta", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Delta.pdf')
    plot_surf(T, gamma_bin_call, "Gamma", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Gamma.pdf')
    plot_surf(T, theta_bin_call, "Theta", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Theta.pdf')
    plot_surf(T, speed_bin_call, "Speed", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Speed.pdf')
    plot_surf(T, vega_bin_call, "Vega", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Vega.pdf')
    plot_surf(T, rho_r_bin_call, "Rho (r)", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Rho_r.pdf')
    plot_surf(T, rho_D_bin_call, "Rho (D)", r'Imagenes/Parte1/6_Sols/Binary_Call/Binary_Call_Rho_D.pdf')

    plot_surf(T, value_bin_put, "Value", r'Imagenes/Parte1/6_Sols/Binary_Put/BinaryPut3D.pdf')
    plot_2D(T, value_bin_put, r'Imagenes/Parte1/6_Sols/Binary_Put/BinaryPutSFijo.pdf', r'Imagenes/Parte1/6_Sols/Binary_Put/BinaryPuttFIjo.pdf')
    plot_surf(T, delta_bin_put, "Delta", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Delta.pdf')
    plot_surf(T, gamma_bin_put, "Gamma", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Gamma.pdf')
    plot_surf(T, theta_bin_put, "Theta", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Theta.pdf')
    plot_surf(T, speed_bin_put, "Speed", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Speed.pdf')
    plot_surf(T, vega_bin_put, "Vega", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Vega.pdf')
    plot_surf(T, rho_r_bin_put, "Rho (r)", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Rho_r.pdf')
    plot_surf(T, rho_D_bin_put, "Rho (D)", r'Imagenes/Parte1/6_Sols/Binary_Put/Binary_Put_Rho_D.pdf')
    
    

    

    

    plt.show()