import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates



if __name__ == "__main__":
    # Parámetros del proceso GBM
    mu = 0.3
    sigma = 0.2
    S0 = 1.5
    T = 2
    N = 500
    dt = T / N
    t = np.linspace(0, T, N + 1)

    # Simulación del GBM
    np.random.seed(2)
    W = np.concatenate(([0], np.cumsum(np.sqrt(dt) * np.random.randn(N))))
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    # Barrera y tiempo de primer cruce
    barrier = 2.5
    first_cross_idx = np.argmax(S >= barrier) if np.any(S >= barrier) else None
    first_cross_time = t[first_cross_idx] if first_cross_idx is not None else None
    first_cross_price = S[first_cross_idx] if first_cross_idx is not None else None

    # Fechas simuladas para los ticks del eje X
    start_date = datetime(1998, 2, 24)
    dates = [start_date + timedelta(days=int(round(x * 365))) for x in t]

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, S, 'b', linewidth=1.5)
    ax.axhline(barrier, color=[0.2, 0.2, 0.2], linewidth=1)
    if first_cross_idx is not None:
        ax.axvline(dates[first_cross_idx], color='k', linestyle='--')
        ax.text(
            dates[first_cross_idx] + timedelta(days=5),
            barrier - 0.7,
            'First-exit time',
            fontsize=10,
            rotation=-45
        )

    ax.set_ylim([0, max(S) * 1.1])
    ax.set_ylabel('Stock price')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/First-exit_Time.pdf', format="pdf", bbox_inches="tight")
    plt.close()


    plt.show()