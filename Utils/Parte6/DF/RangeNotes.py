import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver


class RangeNotes(OptionSolver):
    """
    Range notes option with optional discrete dividends.
    """

    def __init__(self, r : float = 0.05, L : float = 0.1, 
                 P : float = 100, S_t : float = 20, S_u : float = 30,
                 T : float = 1, S_inf : float = 100,
                 EDE : str = "lognormal", 
                 M : int = 500, N : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the RangeNotes option.
        
        Parameters:
        r (float): Risk-free interest rate.
        L (float): Coupon rate paid at maturity if the underlying asset price stays within the range [S_t, S_u].
        P (float): Notional principal amount.
        S_t (float): Lower bound of the range.
        S_u (float): Upper bound of the range.
        T (float): Time to maturity (in years).
        S_inf (float): Maximum asset price considered in the grid.
        EDE (str): Type of the underlying asset dynamics equation. Currently only 'lognormal' is implemented.
        M (int): Number of time steps in the finite-difference grid.
        N (int): Number of asset price steps in the finite-difference grid.
        theta (float): Theta parameter for the finite-difference scheme.
        **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.

        """
        self.r = r
        self.L = L
        self.P = P
        self.S_t = S_t
        self.S_u = S_u
        self.T = T
        self.S_inf = S_inf
        self.EDE = EDE
        self.M = M
        self.N = N
        self.theta = theta

        if EDE == 'lognormal':
            self.sigma = kwargs.get('sigma', 0.2)

            # PDE coefficients stored as attributes
            # V_t + a(t,S) V_SS + b(t,S) V_S + c(t,S) V + f(t,S) = 0
            self.a = lambda t, S: 0.5 * (self.sigma ** 2) * S**2
            self.b = lambda t, S: self.r * S
            self.c = lambda t, S: -self.r
            self.f = lambda t, S: np.where((self.S_t < S) & (S < self.S_u), self.L, 0.0)
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")
        
    def payoff(self, S):
        return self.P
    
    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S=0 for all time slices.
        It must follow the form:
            V(t,0) = phi(0)*exp(∫_t^T c(u,0) du) + ∫_t^T f(u,0) * exp(∫_u^T c(v,0) dv) du
        '''
        if self.EDE == 'lognormal':
            V[0, :] = self.payoff(0) * np.exp( - self.r * (self.T - t)) + np.where((self.S_t < 0) & (0 < self.S_u), self.L, 0.0)/self.r * (1 - np.exp(-self.r * (self.T - t)))
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")
        return V
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rn = RangeNotes()
    rn.plot_value_at_t(0)

    plt.show()




