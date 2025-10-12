import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver



class CompoundOption(OptionSolver):
    """
    Class for pricing compound options.
    """

    name = "Compound Option"

    def __init__(self, underlying_option_1, underlying_option_2,
                 K1 : float = 5, K2 : float = 5,
                 r : float = 0.05,
                 T : float = 0.5, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal',
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs):
        """
        Initializes the CompoundOption class.

        Parameters:
            underlying_option_1: The first underlying option object. Must have a 'T' attribute for maturity and a 'solve()' method.
            underlying_option_2: The second underlying option object. Must have a 'T' attribute for maturity and a 'solve()' method.
            K1 (float, optional): Strike price for the first underlying option. Default is 5.
            K2 (float, optional): Strike price for the second underlying option. Default is 5.
            r (float, optional): Risk-free interest rate. Default is 0.05.
            T (float, optional): Maturity of the compound option. Default is 0.5.
            S_min (float, optional): Minimum value of the underlying asset price grid. Default is 0.
            S_max (float, optional): Maximum value of the underlying asset price grid. Default is 80.
            EDE (str, optional): Model for the underlying asset dynamics. Default is 'lognormal'.
            N (int, optional): Number of grid points for the asset price. Default is 500.
            M (int, optional): Number of grid points for the time discretization. Default is 500.
            theta (float, optional): Theta parameter for the numerical scheme (e.g., Crank-Nicolson). Default is 0.5.
            **kwargs: Additional keyword arguments.
                sigma (float, optional): Volatility of the underlying asset, required if EDE is 'lognormal'. Default is 0.2.
        """
        if T >= underlying_option_1.T:
            raise ValueError("The maturity of the compound option 1 cannot be greater than or equal to that of the underlying option.")
        if T >= underlying_option_2.T:
            raise ValueError("The maturity of the compound option 2 cannot be greater than or equal to that of the underlying option.")

        underlying_option_1.solve()
        underlying_option_2.solve()
        self.underlying_option_1 = underlying_option_1
        self.underlying_option_2 = underlying_option_2
        self.r = r
        self.K1 = K1
        self.K2 = K2
        self.T = T
        self.S_min = S_min
        self.S_max = S_max
        self.EDE = EDE
        self.N = N
        self.M = M
        self.theta = theta

        if EDE == 'lognormal':
            self.sigma = kwargs.get('sigma', 0.2)

            # PDE coefficients stored as attributes
            # V_t + a(t,S) V_SS + b(t,S) V_S + c(t,S) V + f(t,S) = 0
            self.a = lambda t, S: 0.5 * (self.sigma ** 2) * S**2
            self.b = lambda t, S: self.r * S
            self.c = lambda t, S: -self.r
            self.f = lambda t, S: 0.0
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")
        
    def payoff(self,S):
        # Find closest index
        idxS1 = self.underlying_option_1.find_closest_S_index(S)
        idxt1 = self.underlying_option_1.find_closest_t_index(self.T)
        idxS2 = self.underlying_option_2.find_closest_S_index(S)
        idxt2 = self.underlying_option_2.find_closest_t_index(self.T)
        return np.maximum(np.maximum(
                                    self.underlying_option_1.V[idxS1, idxt1] - self.K1, 
                                    self.underlying_option_2.V[idxS2, idxt2] - self.K2),
                                    0)

    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S=0 for all time slices.
        It must follow the form:
            V(t,0) = phi(0)*exp(∫_t^T c(u,0) du) + ∫_t^T f(u,0) * exp(∫_u^T c(v,0) dv) du
        '''
        if self.EDE == 'lognormal':
            V[0, :] = self.payoff(0) * np.exp(self.r * (self.T - t))
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")
        return V
    
    def get_info(self):
        res = (
            f'{self.name}\n'
            f'- Risk-free Rate r: {self.r}\n'
            f'- Strike Prices K1, K2: {self.K1}, {self.K2}\n'
        )
        if self.EDE == 'lognormal':
            res += f'- Volatility sigma: {self.sigma}\n'
        res += self.underlying_option_1.get_info() + "\n"
        res += self.underlying_option_2.get_info()
        return res
    




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean

    T_Ch = 0.5
    T = 1

    call = CallEuropean(T = T)
    put = PutEuropean(T = T)

    chooser = CompoundOption(call, put, K1 = 5, K2 = 5, T = T_Ch)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    chooser.plot_value_at_t(T_Ch, fig=fig, ax=ax, label="Chooser Call Option")
    call.plot_value_at_t(T_Ch, fig=fig, ax=ax, label="Call Option", print_text=False)
    put.plot_value_at_t(T_Ch, fig=fig, ax=ax, label="Put Option", print_text=False)

    plt.show()