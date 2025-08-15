import numpy as np
import matplotlib.pyplot as plt
import argparse



def plot_payoff(payoff_func, S_min, S_max, label=None, axline=True, xlabel="$S(T)$", ylabel="Valor", output_name=None, **kwargs):
    """
    Grafica el payoff dado por una función anónima.

    Parámetros:
    -----------
    payoff_func : function
        Función que calcule el payoff. Ej: lambda S, K: np.maximum(S - K, 0)
    S_min : float
        Precio mínimo del subyacente a graficar.
    S_max : float
        Precio máximo del subyacente a graficar.
    label : str, opcional
        Etiqueta para la curva del payoff.
    axline : bool, opcional
        Si True, dibuja una línea horizontal en y=0.
    xlabel : str, opcional
        Etiqueta para el eje x.
    ylabel : str, opcional
        Etiqueta para el eje y.
    output_name : str, opcional
        Nombre del archivo para guardar la gráfica en formato PDF. Si es None, muestra la gráfica en pantalla.
    **kwargs : 
        Parámetros adicionales que necesite la función payoff_func (ej: E=100)
    """

    S = np.linspace(S_min, S_max, 200)
    payoff = payoff_func(S, **kwargs)
    
    plt.figure()

    # Function and legend
    if label is None:
        plt.plot(S, payoff, color='blue')
    else:
        plt.plot(S, payoff, label=label, color='blue')
        plt.legend()

    # Axes limits
    plt.xlim(S_min, S_max)
    margen = (np.max(payoff)-np.min(payoff))*0.2
    plt.ylim(np.min(payoff)-margen, np.max(payoff)+margen)

    # Axes lines
    if axline:
        plt.axhline(0, color='black', linewidth=0.75, linestyle='--')

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        

    plt.grid(True)

    # Save or show the plot
    if output_name:
        plt.savefig(output_name, format="pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grafique distintos tipos de payoff.")
    parser.add_argument("-p", "--output_path", default=None, help="Opcional: ruta de salida para guardar la gráfica.")

    args = parser.parse_args()
    output_path = args.output_path


    #------------------------------------------------------------------------------
    #PAYOFFS
    #------------------------------------------------------------------------------

    # Call
    call_payoff = lambda S, E: np.maximum(S - E, 0)
    plot_payoff(call_payoff, S_min=50, S_max=150, label=r"$C(S,T)=\max(S-E,0)$", E=100, output_name=r"Imagenes/Parte1/2_Derivados/PayOffCall.pdf")

    # Put
    put_payoff = lambda S, E: np.maximum(E - S, 0)
    plot_payoff(put_payoff, S_min=50, S_max=150, label=r"$P(S,T)=\max(E-S,0)$", E=100, output_name=r"Imagenes/Parte1/2_Derivados/PayOffPut.pdf")

    #Binary call
    binary_call_payoff = lambda S, E: np.where(S > E, 1, 0)
    plot_payoff(binary_call_payoff, S_min=50, S_max=150, label=r"$C_{bin}(S,T)=1_{S>E}$", E=100, output_name=r"Imagenes/Parte1/2_Derivados/BinaryCall.pdf")

    # Binary put
    binary_put_payoff = lambda S, E: np.where(S < E, 1, 0)
    plot_payoff(binary_put_payoff, S_min=50, S_max=150, label=r"$P_{bin}(S,T)=1_{S<E}$", E=100, output_name=r"Imagenes/Parte1/2_Derivados/BinaryPut.pdf")


    #------------------------------------------------------------------------------
    # PROFIT DIAGRAMS
    #------------------------------------------------------------------------------

    # Call
    call_payoff = lambda S, E, prima: np.maximum(S - E, 0) - prima
    plot_payoff(call_payoff, S_min=50, S_max=150, label=r"$C(S,T)=\max(S-E,0)-\pi$", E=100, prima=10, output_name=r'Imagenes/Parte1/2_Derivados/ProfitDiagCall.pdf')

    # Put
    put_payoff = lambda S, E, prima: np.maximum(E - S, 0) - prima
    plot_payoff(put_payoff, S_min=50, S_max=150, label=r"$P(S,T)=\max(E-S,0)-\pi$", E=100, prima=10, output_name=r'Imagenes/Parte1/2_Derivados/ProfitDiagPut.pdf')



    #------------------------------------------------------------------------------
    # OPTION STRATEGIES
    #------------------------------------------------------------------------------

    # 
    # Bull spread
    bull_spread_payoff = lambda S, E1, E2: np.maximum(S - E1, 0) - np.maximum(S - E2, 0)
    plot_payoff(bull_spread_payoff, S_min=50, S_max=150, label=r"$C_{bull}(S,T)=\max(S-E_1,0)-\max(S-E_2,0)$", E1=100, E2=120, output_name=r'Imagenes/Parte1/2_Derivados/BullPayoff.pdf')

    # Bear spread
    bear_spread_payoff = lambda S, E1, E2: np.maximum(E2 - S, 0) - np.maximum(E1 - S, 0)
    plot_payoff(bear_spread_payoff, S_min=50, S_max=150, label=r"$C_{bear}(S,T)=\max(E_2-S,0)-\max(E_1-S,0)$", E1=100, E2=120, output_name=r'Imagenes/Parte1/2_Derivados/BearPayoff.pdf')

    # Straddle
    straddle_payoff = lambda S, E: np.maximum(S - E, 0) + np.maximum(E - S, 0)
    plot_payoff(straddle_payoff, S_min=50, S_max=150, label=r"$C_{straddle}(S,T)=\max(S-E,0)+\max(E-S,0)$", E=100, output_name=r'Imagenes/Parte1/2_Derivados/StraddlePayoff.pdf')

    # Strangle OTM
    strangle_otm_payoff = lambda S, E1, E2: np.maximum(S - E2, 0) + np.maximum(E1 - S, 0)
    plot_payoff(strangle_otm_payoff, S_min=50, S_max=150, label=r"$C_{strangle}(S,T)=\max(S-E_2,0)+\max(E_1-S,0)$", E1=100, E2=120, output_name=r'Imagenes/Parte1/2_Derivados/StrangleOTMPayoff.pdf')

    # Strangle ITM
    strangle_itm_payoff = lambda S, E1, E2: np.maximum(E2 - S, 0) + np.maximum(S - E1, 0)
    plot_payoff(strangle_itm_payoff, S_min=50, S_max=150, label=r"$C_{strangle}(S,T)=\max(E_2-S,0)+\max(S-E_1,0)$", E1=100, E2=120, output_name=r'Imagenes/Parte1/2_Derivados/StrangleITMPayoff.pdf')

    # Bullish risk reversal
    bullish_risk_reversal_payoff = lambda S, Ec, Ep: np.maximum(S - Ec, 0) - np.maximum(Ep - S, 0)
    plot_payoff(bullish_risk_reversal_payoff, S_min=50, S_max=150, label=r"$C_{bullish\ risk\ reversal}(S,T)=\max(S-E_c,0)-\max(E_p-S,0)$", Ec=110, Ep=90, output_name=r'Imagenes/Parte1/2_Derivados/BullishRiskReversalPayoff.pdf')

    # Bearish risk reversal
    bearish_risk_reversal_payoff = lambda S, Ec, Ep: np.maximum(Ep - S, 0) - np.maximum(S - Ec, 0)
    plot_payoff(bearish_risk_reversal_payoff, S_min=50, S_max=150, label=r"$C_{bearish\ risk\ reversal}(S,T)=\max(E_p-S,0)-\max(S-E_c,0)$", Ec=110, Ep=90, output_name=r'Imagenes/Parte1/2_Derivados/BearishRiskReversalPayoff.pdf')

    # Butterfly
    butterfly_payoff = lambda S, Eitm, Eatm, Eotm: np.maximum(S - Eitm, 0) - 2*np.maximum(S - Eatm, 0) + np.maximum(S - Eotm, 0)
    plot_payoff(butterfly_payoff, S_min=50, S_max=150, Eitm=80, Eatm=100, Eotm=120, output_name=r'Imagenes/Parte1/2_Derivados/ButterflyPayoff.pdf')

    # Condors
    condor_payoff = lambda S, E1, E2, E3, E4: np.maximum(S - E1, 0) - np.maximum(S - E2, 0) - np.maximum(S - E3, 0) + np.maximum(S - E4, 0)
    plot_payoff(condor_payoff, S_min=50, S_max=150, E1=80, E2=90, E3=110, E4=120, output_name=r'Imagenes/Parte1/2_Derivados/CondorsPayoff.pdf')








    plt.show()



