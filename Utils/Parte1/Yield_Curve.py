import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza diferentes tipos de curvas de yield.")
    parser.add_argument("-s", "--save", action="store_true", default=False, help="Opcional: guarda el fichero si se activa el flag.")

    args = parser.parse_args()
    save_plot = args.save

    # Datos de plazos (años)
    plazos = np.array([0, 1, 2, 3, 5, 10, 20, 30])

    # Curva normal (creciente)
    yield_normal = np.array([2, 2.5, 2.8, 3.2, 3.5, 3.8, 4, 4.1])

    # Curva plana
    yield_flat = np.full_like(plazos, 3.0, dtype=float)

    # Curva invertida (decreciente)
    yield_inverted = np.array([4, 3.5, 3.2, 2.9, 2.6, 2.4, 2.1, 2.0])

    plt.figure(figsize=(8, 6))
    plt.plot(plazos, yield_normal, color='blue', linewidth=2, label='Normal')
    plt.plot(plazos, yield_flat, color='lime', linewidth=2, label='Flat')
    plt.plot(plazos, yield_inverted, color='red', linewidth=2, label='Inverted')

    plt.xlabel('Plazo (años)', fontsize=12)
    plt.ylabel('Yield', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(1.5, 4.5)
    plt.xlim(0, 30)
    plt.tight_layout()
    plt.savefig(r'Imagenes/Parte1/11_Prods_renta_fija/Yield_Curve.pdf', format="pdf", bbox_inches="tight")
    plt.close()


