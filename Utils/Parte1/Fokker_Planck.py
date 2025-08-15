import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Parámetros del proceso
    mu = 0.05          # Tasa de drift
    sigma = 0.2        # Volatilidad
    t = 0              # Tiempo inicial
    S = 50           # Valor inicial S(t)

    # Rango de valores para S' y t'
    S_prime = np.linspace(0.01, 2*S, 200)         # Eje S'
    t_prime = np.linspace(t + 0.01, 1, 200)     # Eje t' (> t)
    S_grid, T_grid = np.meshgrid(S_prime, t_prime)
    tau = T_grid - t                           # t' - t

    # Solución de la densidad de transición
    p = (1 / (sigma * S_grid * np.sqrt(2 * np.pi * tau))) * \
        np.exp(- (np.log(S / S_grid) + (mu - 0.5 * sigma**2) * tau)**2 / \
        (2 * sigma**2 * tau))

    # === FIGURA 3D ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid, p, cmap='viridis')
    ax.set_xlabel("S'")
    ax.set_ylabel("t'")
    ax.set_title(f"p(S={S}, t={t}; S', t' | μ={mu}, σ={sigma})")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/PDF_3D.pdf', format="pdf", bbox_inches="tight")
    plt.close()


    # === FIGURA 2D para un t' fijo ===
    t_fixed = 0.5
    tau_fixed = t_fixed - t
    p_fixed = (1 / (sigma * S_prime * np.sqrt(2 * np.pi * tau_fixed))) * \
        np.exp(- (np.log(S / S_prime) + (mu - 0.5 * sigma**2) * tau_fixed)**2 / \
        (2 * sigma**2 * tau_fixed))

    plt.figure()
    plt.plot(S_prime, p_fixed, linewidth=2)
    plt.xlabel("S'")
    plt.title(f"p(S={S}, t={t}; S', t'={t_fixed} | μ={mu}, σ={sigma})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'Imagenes/Parte1/3_Aleatoriedad/PDF_2D.pdf', format="pdf", bbox_inches="tight")
    plt.close()



    plt.show()