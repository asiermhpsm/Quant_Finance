import numpy as np

import matplotlib.pyplot as plt



if __name__ == "__main__":



    
    # ================================
    # Simulación y estimación de volatilidad
    # ================================

    # Parámetros de simulación
    N = 300                      # Número de días
    mu = 0.0005                  # Retorno promedio diario
    sigma_true = 0.2             # Volatilidad diaria real
    np.random.seed(1)

    # Simulación de precios
    log_ret = mu + sigma_true * np.random.randn(N)
    prices = 100 * np.exp(np.cumsum(log_ret))  # Precio inicial = 100

    # Cálculo de retornos diarios
    returns = np.diff(prices) / prices[:-1]
    returns = returns.reshape(-1, 1)  # Asegurar columna
    n_ret = len(returns)

    # === Estimaciones de volatilidad ===

    # 1. Media histórica (volatilidad constante)
    vol_const = np.sqrt(np.mean(returns ** 2)) * np.ones((n_ret, 1))

    # 2. ARCH: Regresión a la media
    alpha_arch = 0.2
    sigma_bar = np.sqrt(np.mean(returns ** 2))  # estimación de largo plazo
    window_arch = 20
    vol_arch = np.zeros((n_ret, 1))
    for i in range(window_arch, n_ret):
        recent_vol = np.mean(returns[i - window_arch + 1:i + 1] ** 2)
        vol_arch[i] = np.sqrt(alpha_arch * sigma_bar ** 2 + (1 - alpha_arch) * recent_vol)

    # 3. EWMA
    lambda_ewma = 0.94
    vol_ewma = np.zeros((n_ret, 1))
    vol_ewma[0] = np.sqrt(returns[0] ** 2)  # inicialización
    for i in range(1, n_ret):
        vol_ewma[i] = np.sqrt(lambda_ewma * vol_ewma[i - 1] ** 2 + (1 - lambda_ewma) * returns[i] ** 2)

    # 4. GARCH-like
    alpha_garch = 0.1
    lambda_garch = 0.9
    vol_garch = np.zeros((n_ret, 1))
    vol_garch[0] = np.sqrt(sigma_bar ** 2)
    for i in range(1, n_ret):
        vol_garch[i] = np.sqrt(alpha_garch * sigma_bar ** 2 + (1 - alpha_garch) * (lambda_garch * vol_garch[i - 1] ** 2 + (1 - lambda_garch) * returns[i] ** 2))

    # === Línea de la solución exacta (volatilidad real)
    vol_exact = sigma_true * np.ones((n_ret, 1))

    # === Gráfico 1: Volatilidades ===
    t = np.arange(2, N + 1)  # Tiempos para los retornos

    plt.figure()
    plt.plot(t, vol_exact, 'k--', linewidth=2, label='Solución exacta')
    plt.plot(t, vol_const, 'm-', label='Constante (Histórica)')
    plt.plot(t, vol_arch, 'b-', label='ARCH')
    plt.plot(t, vol_ewma, 'r-', label='EWMA')
    plt.plot(t, vol_garch, 'g-', label='GARCH-like')
    plt.xlabel('Día')
    plt.ylabel('Volatilidad diaria estimada')
    plt.ylim([0.1, 0.3])
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(r'Imagenes/Parte1/7_Volatilidad/Volatilidad.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    # === Gráfico 2: Evolución del precio ===
    plt.figure()
    plt.plot(np.arange(1, N + 1), prices, linewidth=1.5)
    plt.xlabel('Día')
    plt.ylabel('Precio simulado')
    plt.grid(True)
    plt.savefig(r'Imagenes/Parte1/7_Volatilidad/Accion.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    # === Estimación de volatilidad futura esperada (extensión de las series) ===

    k_max = 100  # horizonte de días hacia el futuro
    future_days = np.arange(n_ret + 1, n_ret + k_max + 1)

    # Últimas volatilidades estimadas (día n)
    sigma_n2_ewma = vol_ewma[-1] ** 2
    sigma_n2_garch = vol_garch[-1] ** 2

    # 1. EWMA futura: constante
    vol_ewma_extended = np.vstack([vol_ewma, np.sqrt(sigma_n2_ewma) * np.ones((k_max, 1))])

    # 2. GARCH-like futura: convergencia a sigma_bar
    nu = alpha_garch / (1 - (1 - alpha_garch) * (1 - lambda_garch))
    garch_future2 = sigma_bar ** 2 + (sigma_n2_garch - sigma_bar ** 2) * (1 - nu) ** np.arange(1, k_max + 1)
    vol_garch_extended = np.vstack([vol_garch, np.sqrt(garch_future2).reshape(-1, 1)])

    # Línea de referencia: volatilidad real constante
    vol_exact_extended = np.vstack([vol_exact, sigma_true * np.ones((k_max, 1))])
    t_total = np.arange(2, N + k_max + 1)

    # === Nuevo gráfico: volatilidad estimada + futura con separación visual ===
    plt.figure()
    plt.plot(t_total, vol_exact_extended, 'k--', linewidth=2, label='Solución exacta')

    # Parte observada
    plt.plot(np.arange(2, N + 1), vol_ewma, 'r-', label='EWMA (estimado)')
    plt.plot(np.arange(2, N + 1), vol_garch, 'g-', label='GARCH-like (estimado)')

    # Parte futura (usar n_ret + 1 como día inicial)
    plt.plot(np.arange(n_ret + 1, n_ret + k_max + 1), vol_ewma_extended[n_ret:], 'r--', label='EWMA (futuro)')
    plt.plot(np.arange(n_ret + 1, n_ret + k_max + 1), vol_garch_extended[n_ret:], 'g--', label='GARCH-like (futuro)')

    # Línea vertical en el corte
    plt.axvline(n_ret, color='k', linestyle=':', linewidth=1, label='Inicio predicción')

    plt.xlabel('Día')
    plt.ylabel('Volatilidad diaria estimada')
    plt.ylim([0.1, 0.3])
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(r'Imagenes/Parte1/7_Volatilidad/Volatilidad_Prediccion.pdf', format="pdf", bbox_inches="tight")
    plt.close()