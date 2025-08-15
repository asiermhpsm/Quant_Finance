import random
import matplotlib.pyplot as plt
import numpy as np

def generate_dW(T, N, random_seed=None):
    """
    Genera incrementos dW para un proceso de Wiener.

    Parámetros:
    -----------
    T : float
        Tiempo total.
    N : int
        Número de pasos.
    random_seed : int, opcional
        Semilla para la generación aleatoria.

    Retorna:
    --------
    dW : ndarray
        Incrementos del proceso Wiener.
    dt : float
        Paso temporal.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = T / N
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
    return dW, dt

def brownian_motion_with_drift(T, N, S0=1, mu=0, sigma=1, random_seed=None):
    """
    Camino browniano con deriva: dS = mu*dt + sigma*dW
    """
    dW, dt = generate_dW(T, N, random_seed)
    t = np.linspace(0, T, N + 1)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i-1] +  mu * dt + sigma * dW[i-1]
    return t, S

def lognormal_random_walk(T, N, S0=1, mu=0, sigma=1, random_seed=None):
    """
    Camino lognormal: dS = S*mu*dt + S*sigma*dW.
    """
    dW, dt = generate_dW(T, N, random_seed)
    t = np.linspace(0, T, N + 1)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i-1] + S[i-1] * mu * dt + S[i-1] * sigma * dW[i-1]
    return t, S

def mean_reverting_random_walk(T, N, S0=1, kappa=0, theta=0.5, sigma=1, random_seed=None):
    """
    Camino aleatorio mean-reverting: dS = kappa*(theta-S)*dt + sigma*dW.
    """
    dW, dt = generate_dW(T, N, random_seed)
    t = np.linspace(0, T, N + 1)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i-1] + kappa * (theta - S[i-1]) * dt + sigma * dW[i-1]
    return t, S


def plot_random_walk(t, S, series_label, fig=None, ax=None):
    """
    Crea una gráfica para un camino aleatorio y devuelve la figura.

    Parámetros:
    -----------
    t : array
        Array de tiempo.
    S : array
        Array de precios del activo.
    series_label : str
        Etiqueta para la serie de datos.
    fig : matplotlib.figure.Figure
        Figura a utilizar para la gráfica.
    ax : matplotlib.axes.Axes
        Ejes a utilizar para la gráfica.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, label=series_label)
    ax.axhline(0, color='black', linewidth=0.75, linestyle='--')

    ax.set_xlabel("t")
    ax.set_ylabel("Valor")

    ax.grid(True)
    ax.legend()

    return fig, ax


if __name__ == "__main__":
    # --- Parámetros para las simulaciones ---
    T = 10  # Tiempo total (años)
    N = 365*T  # Número de pasos (días de trading en un año)
    S0 = 1 # Precio inicial
    mu = 0.2 # Deriva/Rendimiento esperado
    sigma = 0.3 # Volatilidad
    kappa = mu
    theta = 5 
    random_seed = 50 # Semilla para reproducibilidad


    # 1. Camino Browniano con deriva
    t, S = brownian_motion_with_drift(T=T, N=N, S0=S0, mu=mu, sigma=sigma, random_seed=random_seed)
    fig, ax = plot_random_walk(
        t, 
        S, 
        series_label=f'μ={mu}, σ={sigma}'
    )
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/BrownianMotionDrift.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    # 2. Camino Lognormal
    t, S = lognormal_random_walk(T=T, N=N, S0=S0, mu=mu, sigma=sigma, random_seed=random_seed)
    fig, ax = plot_random_walk(
        t, 
        S, 
        series_label=f'μ={mu}, σ={sigma}'
    )
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/LognormalRandomWalk.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    # 3. Camino con Reversión a la Media
    t, S = mean_reverting_random_walk(T=T, N=N, S0=S0, kappa=kappa, theta=theta, sigma=sigma, random_seed=random_seed)
    fig, ax = plot_random_walk(
        t, 
        S, 
        series_label=f'κ={kappa}, θ={theta}, σ={sigma}',
    )

    t, S = mean_reverting_random_walk(T=T, N=N, S0=S0, kappa=kappa, theta=theta, sigma=0, random_seed=random_seed)
    fig, ax = plot_random_walk(
        t, 
        S, 
        series_label=f'Media',
        fig=fig,
        ax=ax
    )
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/MeanRevertingWalk.pdf', format="pdf", bbox_inches="tight")
    plt.close()



    plt.show()