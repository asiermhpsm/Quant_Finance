import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver

class CompoundOption(OptionSolver):
    """
    Class for pricing compound options.
    """

    name = "Compound Option"

    def __init__(self, underlying_option, r : float = 0.05, K : float = 20,
                 T : float = 0.5, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal',
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs):
        """
        Initializes the CompoundOption class.

        Parameters:
            r (float): Risk-free interest rate.
            K (float): Strike price of the underlying option.
            T (float): Time to maturity of the underlying option.
            S_init (float): Initial stock price.
            EDE (str): Stochastic process for the underlying asset ('lognormal').
            N (int): Number of time steps steps in the grid.
            M (int): Number of stock steps in the grid.
            theta (float): Implicitness parameter for the finite difference method.
            **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.
        """
        if T >= underlying_option.T:
            raise ValueError("The maturity of the compound option cannot be greater than or equal to that of the underlying option.")

        underlying_option.solve()
        self.underlying_option = underlying_option
        self.r = r
        self.K = K
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
        
    def get_info(self):
        res = (
            f'{self.name}\n'
            f'- Strike Price K: {self.K}\n'
            f'- Risk-free Rate r: {self.r}'
        )
        if self.EDE == 'lognormal':
            res += f'\n- Volatility sigma: {self.sigma}'
        return res
        


class CallCompoundOption(CompoundOption):
    """
    Class for pricing call compound options.
    """

    name = "Call Compound Option"

    def payoff(self,S):
        # Find closest index
        idxS = self.underlying_option.find_closest_S_index(S)
        idxt = self.underlying_option.find_closest_t_index(self.T)
        return np.maximum(self.underlying_option.V[idxS, idxt] - self.K, 0)

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

class PutCompoundOption(CompoundOption):
    """
    Class for pricing put compound options.
    """

    name = "Put Compound Option"

    def payoff(self,S):
        # Find closest index
        idxS = self.underlying_option.find_closest_S_index(S)
        idxt = self.underlying_option.find_closest_t_index(self.T)
        return np.maximum(self.K - self.underlying_option.V[idxS, idxt], 0)

    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S=0 for all time slices.
        It must follow the form:
            V(t,0) = phi(0)*exp(∫_t^T c(u,0) du) + ∫_t^T f(u,0) * exp(∫_u^T c(v,0) dv) du
        '''
        if self.EDE == 'lognormal':
            V[0, :] = self.payoff(0) * np.exp(self.r * (self.T - t))
            return V
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")

class BinaryCallCompoundOption(CompoundOption):
    """
    Class for pricing binary (cash-or-nothing) call compound options.
    """

    name = "Binary Call Compound Option"

    def payoff(self,S):
        # Find closest index
        idxS = self.underlying_option.find_closest_S_index(S)
        idxt = self.underlying_option.find_closest_t_index(self.T)
        return np.where(self.underlying_option.V[idxS, idxt] > self.K, 1.0, 0.0)

    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S=0 for all time slices.
        It must follow the form:
            V(t,0) = phi(0)*exp(∫_t^T c(u,0) du) + ∫_t^T f(u,0) * exp(∫_u^T c(v,0) dv) du
        '''
        if self.EDE == 'lognormal':
            V[0, :] = self.payoff(0) * np.exp(self.r * (self.T - t))
            return V
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")

class BinaryPutCompoundOption(CompoundOption):
    """
    Class for pricing binary (cash-or-nothing) put compound options.
    """

    name = "Binary Put Compound Option"

    def payoff(self,S):
        # Find closest index
        idxS = self.underlying_option.find_closest_S_index(S)
        idxt = self.underlying_option.find_closest_t_index(self.T)
        return np.where(self.underlying_option.V[idxS, idxt] < self.K, 1.0, 0.0)

    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S=0 for all time slices.
        It must follow the form:
            V(t,0) = phi(0)*exp(∫_t^T c(u,0) du) + ∫_t^T f(u,0) * exp(∫_u^T c(v,0) dv) du
        '''
        if self.EDE == 'lognormal':
            V[0, :] = self.payoff(0) * np.exp(self.r * (self.T - t))
            return V
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean, BinaryCallEuropean, BinaryPutEuropean

    T_C0 = 0.5
    T = 1

    print("Generating underlying options...")
    print("Generating Call Option...")
    call = CallEuropean(T = T)
    print("Generating Put Option...")
    put = PutEuropean(T = T)
    print("Generating Binary Call Option...")
    binary_call = BinaryCallEuropean(T = T)
    print("Generating Binary Put Option...")
    binary_put = BinaryPutEuropean(T = T)

    # Call on call
    print("Testing Call on Call Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_call = CallCompoundOption(call, T=T_C0)
    call_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    call.plot_value_at_t(0, fig=fig, ax=ax, label="Call Option")
    ax.set_title("Compound Call Option on Call Option")

    # Call on put
    print("Testing Call on Put Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_put = CallCompoundOption(put, T=T_C0)
    call_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    put.plot_value_at_t(0, fig=fig, ax=ax, label="Put Option")
    ax.set_title("Compound Call Option on Put Option")

    # Call on binary call
    print("Testing Call on Binary Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_binary_call = CallCompoundOption(binary_call, T=T_C0)
    call_binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call Option")
    ax.set_title("Compound Call Option on Binary Call Option")

    # Call on binary put
    print("Testing Call on Binary Put Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_binary_put = CallCompoundOption(binary_put, T=T_C0)
    call_binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put Option")
    ax.set_title("Compound Call Option on Binary Put Option")

    # Put on call
    print("Testing Put on Call Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_call = PutCompoundOption(call, T=T_C0)
    put_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    call.plot_value_at_t(0, fig=fig, ax=ax, label="Call Option")
    ax.set_title("Compound Put Option on Call Option")

    # Put on put
    print("Testing Put on Put Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_put = PutCompoundOption(put, T=T_C0)
    put_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    put.plot_value_at_t(0, fig=fig, ax=ax, label="Put Option")
    ax.set_title("Compound Put Option on Put Option")

    # Put on binary call
    print("Testing Put on Binary Call Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_binary_call = PutCompoundOption(binary_call, T=T_C0)
    put_binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call Option")
    ax.set_title("Compound Put Option on Binary Call Option")

    # Put on binary put
    print("Testing Put on Binary Put Compound Options")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_binary_put = PutCompoundOption(binary_put, T=T_C0)
    put_binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put Option")
    ax.set_title("Compound Put Option on Binary Put Option")

    

    plt.show()