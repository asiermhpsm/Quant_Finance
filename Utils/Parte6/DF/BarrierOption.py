import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver



# ----------------------------------------------------------------------------------------------------------------
# Up and Out Barrier Options
# ----------------------------------------------------------------------------------------------------------------

class UpAndOutOption(OptionSolver):
    def __init__(self, r : float = 0.05, S_u : float = 50,
                 T : float = 1, S_inf : float = 100,
                 EDE : str = 'lognormal', 
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the European option with given parameters.

        Parameters:
            r (float): Risk-free interest rate.
            K (float): Strike price.
            D (float): Continuous dividend yield.
            T (float): Time to maturity.
            S_inf (float): Maximum stock price considered in the grid.
            EDE (str): Stochastic process for the underlying asset ('lognormal').
            N (int): Number of time steps steps in the grid.
            M (int): Number of stock steps in the grid.
            theta (float): Parameter for the theta-method in finite difference scheme.
            **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.
        """


