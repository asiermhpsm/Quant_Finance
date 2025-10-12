import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver
from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean, BinaryCallEuropean, BinaryPutEuropean




# ----------------------------------------------------------------------------------------------------------------
# Up and Out Barrier Options
# ----------------------------------------------------------------------------------------------------------------

class UpAndOutOption(OptionSolver):

    name = "Up and Out Barrier Option"

    def __init__(self, r : float = 0.05, K: float = 20, S_u : float = 40, R : float = 0,
                 T : float = 1, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal', 
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the European option with given parameters.

        Parameters:
            r (float): Risk-free interest rate.
            K (float): Strike price.
            S_u (float): Barrier level.
            R (float): Rebate paid when the barrier is hit.
            T (float): Time to maturity.
            S_min (float): Minimum stock price in the grid.
            S_max (float): Maximum stock price in the grid.
            EDE (str): Stochastic process for the underlying asset ('lognormal').
            N (int): Number of time steps steps in the grid.
            M (int): Number of stock steps in the grid.
            theta (float): Parameter for the theta-method in finite difference scheme.
            **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.
        """

        self.r = r
        self.K = K
        self.S_u = S_u
        self.R = R
        self.T = T
        self.S_max = S_max
        self.S_min = S_min
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
        
    def solve(self, known_boundaries: str = None):
        # Save original parameters
        S_max_full = self.S_max
        M_full = self.M
        S_min = self.S_min
        S_u = self.S_u

        # Adjust new parameters to solve in [S_min, S_u]
        self.S_max = S_u
        self.M = int(M_full * (S_u - S_min) / (S_max_full - S_min))

        # Solve
        super().solve(known_boundaries=known_boundaries)

        # Expand solution to original grid
        S_inner, t_inner, V_inner = self.S, self.t, self.V

        S_full = np.linspace(S_min, S_max_full, M_full + 2)
        t_axis = t_inner[0, :]
        S_mesh_full, t_mesh_full = np.meshgrid(S_full, t_axis, indexing="ij")

        V_full = np.full((M_full + 2, t_axis.size), self.R, dtype=float)
        V_full[0:(self.M + 2), :] = V_inner

        # Restore original parameters
        self.S_max = S_max_full
        self.M = M_full
        self.S = S_mesh_full
        self.t = t_mesh_full
        self.V = V_full

    def apply_Smax(self, t, V):
        '''
        Apply the boundary condition at S=S_max for all time slices.

        '''
        V[-1, :] = self.R
        return V
    
    def get_info(self):
        res = (
            f'{self.name}\n'
            f'- Risk-free Rate r: {self.r}\n'
            f'- Strike K: {self.K}\n'
            f'- Rebate R: {self.R}\n'
            f'- Barrier: {self.S_u} '
        )
        if self.EDE == 'lognormal':
            res += f'\n- Volatility sigma: {self.sigma}'
        return res   


class CallUpAndOutOption(UpAndOutOption):

    name = "Call Up and Out Barrier Option"

    def payoff(self, S):
        return np.maximum(S - self.K, 0)
    
class PutUpAndOutOption(UpAndOutOption):

    name = "Put Up and Out Barrier Option"

    def payoff(self, S):
        return np.maximum(self.K - S, 0)
    
class BinaryCallUpAndOutOption(UpAndOutOption):

    name = "Binary Call Up and Out Barrier Option"

    def payoff(self, S):
        return np.where(S < self.K, 0, 1)
    
class BinaryPutUpAndOutOption(UpAndOutOption):

    name = "Binary Put Up and Out Barrier Option"

    def payoff(self, S):
        return np.where(S > self.K, 0, 1)


# ----------------------------------------------------------------------------------------------------------------
# Down and Out Barrier Options
# ----------------------------------------------------------------------------------------------------------------

class DownAndOutOption(OptionSolver):

    name = "Down and Out Barrier Option"

    def __init__(self, r : float = 0.05, K: float = 20, S_d : float = 40, R : float = 0,
                 T : float = 1, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal', 
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the European option with given parameters.

        Parameters:
            r (float): Risk-free interest rate.
            K (float): Strike price.
            S_d (float): Barrier level.
            R (float): Rebate paid when the barrier is hit.
            T (float): Time to maturity.
            S_min (float): Minimum stock price in the grid.
            S_max (float): Maximum stock price in the grid.
            EDE (str): Stochastic process for the underlying asset ('lognormal').
            N (int): Number of time steps steps in the grid.
            M (int): Number of stock steps in the grid.
            theta (float): Parameter for the theta-method in finite difference scheme.
            **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.
        """

        self.r = r
        self.K = K
        self.S_d = S_d
        self.R = R
        self.T = T
        self.S_max = S_max
        self.S_min = S_min
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
        
    def solve(self, known_boundaries: str = None):
        # Save original parameters
        S_min_full = self.S_min
        S_max_full = self.S_max
        M_full = self.M
        S_d = self.S_d

        # Adjust new parameters to solve in [S_d, S_max]
        self.S_min = S_d
        self.M = int(M_full * (S_max_full - S_d) / (S_max_full - S_min_full))

        # Solve
        super().solve(known_boundaries=known_boundaries)

        # Expand solution to original grid
        S_inner, t_inner, V_inner = self.S, self.t, self.V

        S_full = np.linspace(S_min_full, S_max_full, M_full + 2)
        t_axis = t_inner[0, :]
        S_mesh_full, t_mesh_full = np.meshgrid(S_full, t_axis, indexing="ij")

        V_full = np.full((M_full + 2, t_axis.size), self.R, dtype=float)
        V_full[-(self.M + 2):, :] = V_inner  # place inner solution from S_d to S_max

        # Restore original parameters
        self.S_min = S_min_full
        self.M = M_full
        self.S = S_mesh_full
        self.t = t_mesh_full
        self.V = V_full

    def apply_S0(self, t, V):
        '''
        Apply the boundary condition at S = S_d (lower barrier) for all time slices.
        '''
        V[0, :] = self.R
        return V
    
    def get_info(self):
        res = (
            f'{self.name}\n'
            f'- Risk-free Rate r: {self.r}\n'
            f'- Strike K: {self.K}\n'
            f'- Rebate R: {self.R}\n'
            f'- Barrier: {self.S_d} '
        )
        if self.EDE == 'lognormal':
            res += f'\n- Volatility sigma: {self.sigma}'
        return res   


class CallDownAndOutOption(DownAndOutOption):

    name = "Call Down and Out Barrier Option"

    def payoff(self, S):
        return np.maximum(S - self.K, 0)
    
class PutDownAndOutOption(DownAndOutOption):

    name = "Put Down and Out Barrier Option"

    def payoff(self, S):
        return np.maximum(self.K - S, 0)
    
class BinaryCallDownAndOutOption(DownAndOutOption):

    name = "Binary Call Down and Out Barrier Option"

    def payoff(self, S):
        return np.where(S < self.K, 0, 1)
    
class BinaryPutDownAndOutOption(DownAndOutOption):

    name = "Binary Put Down and Out Barrier Option"

    def payoff(self, S):
        return np.where(S > self.K, 0, 1)


# ----------------------------------------------------------------------------------------------------------------
# Up and In Barrier Options
# ----------------------------------------------------------------------------------------------------------------

class UpAndInOption(OptionSolver):

    name = "Up and In Barrier Option"

    def __init__(self, r : float = 0.05, K: float = 20, S_u : float = 40, R : float = 0,
                 T : float = 1, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal', 
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the European option with given parameters.

        Parameters:
            r (float): Risk-free interest rate.
            K (float): Strike price.
            S_u (float): Barrier level.
            R (float): Rebate paid when the barrier is hit.
            T (float): Time to maturity.
            S_min (float): Minimum stock price in the grid.
            S_max (float): Maximum stock price in the grid.
            EDE (str): Stochastic process for the underlying asset ('lognormal').
            N (int): Number of time steps steps in the grid.
            M (int): Number of stock steps in the grid.
            theta (float): Parameter for the theta-method in finite difference scheme.
            **kwargs: Additional keyword arguments:
                if EDE is 'lognormal':
                    sigma (float): Volatility of the underlying asset.
        """

        self.r = r
        self.K = K
        self.S_u = S_u
        self.R = R
        self.T = T
        self.S_max = S_max
        self.S_min = S_min
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
            f'- Risk-free Rate r: {self.r}\n'
            f'- Strike K: {self.K}\n'
            f'- Rebate R: {self.R}\n'
            f'- Barrier: {self.S_u} '
        )
        if self.EDE == 'lognormal':
            res += f'\n- Volatility sigma: {self.sigma}'
        return res   


class CallUpAndInOption(UpAndInOption):

    name = "Call Up and In Barrier Option"

    def solve(self):
        call_out = CallUpAndOutOption(r=self.r, K=self.K, S_u=self.S_u, R=self.R,
                                     T=self.T, S_min=self.S_min, S_max=self.S_max,
                                     EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                     sigma=self.sigma)
        S_out, t_out, V_out = call_out.get_solution()

        call_vanilla = CallEuropean(r=self.r, K=self.K, D=0,
                                          T=self.T, S_min=self.S_min, S_max=self.S_max,
                                          EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                          sigma=self.sigma)
        _, _, V_vanilla = call_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class PutUpAndInOption(UpAndInOption):

    name = "Put Up and In Barrier Option"

    def solve(self):
        put_out = PutUpAndOutOption(r=self.r, K=self.K, S_u=self.S_u, R=self.R,
                                    T=self.T, S_min=self.S_min, S_max=self.S_max,
                                    EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                    sigma=self.sigma)
        S_out, t_out, V_out = put_out.get_solution()

        put_vanilla = PutEuropean(r=self.r, K=self.K, D=0,
                                        T=self.T, S_min=self.S_min, S_max=self.S_max,
                                        EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                        sigma=self.sigma)
        _, _, V_vanilla = put_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class BinaryCallUpAndInOption(UpAndInOption):

    name = "Binary Call Up and In Barrier Option"

    def solve(self):
        bin_call_out = BinaryCallUpAndOutOption(r=self.r, K=self.K, S_u=self.S_u, R=self.R,
                                                T=self.T, S_min=self.S_min, S_max=self.S_max,
                                                EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                                sigma=self.sigma)
        S_out, t_out, V_out = bin_call_out.get_solution()

        bin_call_vanilla = BinaryCallEuropean(r=self.r, K=self.K, D=0,
                                                    T=self.T, S_min=self.S_min, S_max=self.S_max,
                                                    EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                                    sigma=self.sigma)
        _, _, V_vanilla = bin_call_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class BinaryPutUpAndInOption(UpAndInOption):

    name = "Binary Put Up and In Barrier Option"

    def solve(self):
        bin_put_out = BinaryPutUpAndOutOption(r=self.r, K=self.K, S_u=self.S_u, R=self.R,
                                              T=self.T, S_min=self.S_min, S_max=self.S_max,
                                              EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                              sigma=self.sigma)
        S_out, t_out, V_out = bin_put_out.get_solution()

        bin_put_vanilla = BinaryPutEuropean(r=self.r, K=self.K, D=0,
                                                  T=self.T, S_min=self.S_min, S_max=self.S_max,
                                                  EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                                  sigma=self.sigma)
        _, _, V_vanilla = bin_put_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out





# ----------------------------------------------------------------------------------------------------------------
# Down and In Barrier Options
# ----------------------------------------------------------------------------------------------------------------

class DownAndInOption(OptionSolver):

    name = "Down and In Barrier Option"

    def __init__(self, r : float = 0.05, K: float = 20, S_d : float = 40, R : float = 0,
                 T : float = 1, 
                 S_min : float = 0, S_max : float = 80,
                 EDE : str = 'lognormal', 
                 N : int = 500, M : int = 500, theta : float = 0.5,
                 **kwargs
                 ):
        """
        Initialize the Down and In option with given parameters.
        """
        self.r = r
        self.K = K
        self.S_d = S_d
        self.R = R
        self.T = T
        self.S_max = S_max
        self.S_min = S_min
        self.EDE = EDE
        self.N = N
        self.M = M
        self.theta = theta

        if EDE == 'lognormal':
            self.sigma = kwargs.get('sigma', 0.2)
            self.a = lambda t, S: 0.5 * (self.sigma ** 2) * S**2
            self.b = lambda t, S: self.r * S
            self.c = lambda t, S: -self.r
            self.f = lambda t, S: 0.0
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")
        
    def get_info(self):
        res = (
            f'{self.name}\n'
            f'- Risk-free Rate r: {self.r}\n'
            f'- Strike K: {self.K}\n'
            f'- Rebate R: {self.R}\n'
            f'- Barrier: {self.S_d} '
        )
        if self.EDE == 'lognormal':
            res += f'\n- Volatility sigma: {self.sigma}'
        return res   


class CallDownAndInOption(DownAndInOption):

    name = "Call Down and In Barrier Option"

    def solve(self):
        call_out = CallDownAndOutOption(r=self.r, K=self.K, S_d=self.S_d, R=self.R,
                                        T=self.T, S_min=self.S_min, S_max=self.S_max,
                                        EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                        sigma=self.sigma)
        S_out, t_out, V_out = call_out.get_solution()

        call_vanilla = CallEuropean(r=self.r, K=self.K, D=0,
                                    T=self.T, S_min=self.S_min, S_max=self.S_max,
                                    EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                    sigma=self.sigma)
        _, _, V_vanilla = call_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class PutDownAndInOption(DownAndInOption):

    name = "Put Down and In Barrier Option"

    def solve(self):
        put_out = PutDownAndOutOption(r=self.r, K=self.K, S_d=self.S_d, R=self.R,
                                      T=self.T, S_min=self.S_min, S_max=self.S_max,
                                      EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                      sigma=self.sigma)
        S_out, t_out, V_out = put_out.get_solution()

        put_vanilla = PutEuropean(r=self.r, K=self.K, D=0,
                                  T=self.T, S_min=self.S_min, S_max=self.S_max,
                                  EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                  sigma=self.sigma)
        _, _, V_vanilla = put_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class BinaryCallDownAndInOption(DownAndInOption):

    name = "Binary Call Down and In Barrier Option"

    def solve(self):
        bin_call_out = BinaryCallDownAndOutOption(r=self.r, K=self.K, S_d=self.S_d, R=self.R,
                                                  T=self.T, S_min=self.S_min, S_max=self.S_max,
                                                  EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                                  sigma=self.sigma)
        S_out, t_out, V_out = bin_call_out.get_solution()

        bin_call_vanilla = BinaryCallEuropean(r=self.r, K=self.K, D=0,
                                              T=self.T, S_min=self.S_min, S_max=self.S_max,
                                              EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                              sigma=self.sigma)
        _, _, V_vanilla = bin_call_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

class BinaryPutDownAndInOption(DownAndInOption):

    name = "Binary Put Down and In Barrier Option"

    def solve(self):
        bin_put_out = BinaryPutDownAndOutOption(r=self.r, K=self.K, S_d=self.S_d, R=self.R,
                                                T=self.T, S_min=self.S_min, S_max=self.S_max,
                                                EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                                sigma=self.sigma)
        S_out, t_out, V_out = bin_put_out.get_solution()

        bin_put_vanilla = BinaryPutEuropean(r=self.r, K=self.K, D=0,
                                            T=self.T, S_min=self.S_min, S_max=self.S_max,
                                            EDE=self.EDE, N=self.N, M=self.M, theta=self.theta,
                                            sigma=self.sigma)
        _, _, V_vanilla = bin_put_vanilla.get_solution()

        self.S = S_out
        self.t = t_out
        self.V = V_vanilla - V_out

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t0 = 0

    call = CallEuropean()
    put = PutEuropean()
    binarycall = BinaryCallEuropean()
    binaryput = BinaryPutEuropean()

    # Call Up
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    call_out = CallUpAndOutOption()
    call_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Call Up and Out")
    call_in = CallUpAndInOption()
    call_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Call Up and In", print_text=False)
    call.plot_value_at_t(t0, fig=fig, ax=ax, label="Call European", alpha=0.35, print_text=False)

    # Call Down
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    call_out = CallDownAndOutOption()
    call_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Call Down and Out")
    call_in = CallDownAndInOption()
    call_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Call Down and In", print_text=False)
    call.plot_value_at_t(t0, fig=fig, ax=ax, label="Call European", alpha=0.35, print_text=False)

    # Put Up
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    put_out = PutUpAndOutOption()
    put_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Put Up and Out")
    put_in = PutUpAndInOption()
    put_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Put Up and In", print_text=False)
    put.plot_value_at_t(t0, fig=fig, ax=ax, label="Put European", alpha=0.35, print_text=False)

    # Put Down
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    put_out = PutDownAndOutOption()
    put_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Put Down and Out")
    put_in = PutDownAndInOption()
    put_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Put Down and In", print_text=False)
    put.plot_value_at_t(t0, fig=fig, ax=ax, label="Put European", alpha=0.35, print_text=False)

    # Binary Call Up
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bcall_out = BinaryCallUpAndOutOption()
    bcall_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call Up and Out")
    bcall_in = BinaryCallUpAndInOption()
    bcall_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call Up and In", print_text=False)
    binarycall.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call European", alpha=0.35, print_text=False)

    # Binary Call Down
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bcall_out = BinaryCallDownAndOutOption()
    bcall_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call Down and Out")
    bcall_in = BinaryCallDownAndInOption()
    bcall_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call Down and In", print_text=False)
    binarycall.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Call European", alpha=0.35, print_text=False)

    # Binary Put Up
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bput_out = BinaryPutUpAndOutOption()
    bput_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put Up and Out")
    bput_in = BinaryPutUpAndInOption()
    bput_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put Up and In", print_text=False)
    binaryput.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put European", alpha=0.35, print_text=False)

    # Binary Put Down
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    bput_out = BinaryPutDownAndOutOption()
    bput_out.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put Down and Out")
    bput_in = BinaryPutDownAndInOption()
    bput_in.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put Down and In", print_text=False)
    binaryput.plot_value_at_t(t0, fig=fig, ax=ax, label="Binary Put European", alpha=0.35, print_text=False)

    plt.show()
