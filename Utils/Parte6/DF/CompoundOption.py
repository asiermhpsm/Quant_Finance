import numpy as np

from Utils.Parte6.DF.OptionSolver import OptionSolver, _thomas_solver

class CompoundOption(OptionSolver):
    """
    Class for pricing compound options.
    """

    def __init__(self, underlying_option, r : float = 0.05, K : float = 20,
                 T : float = 0.5, S_inf : float = 80,
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
        self.S_inf = S_inf
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
        
    def solve(self):
        """
        Solve the option pricing PDE using a finite-difference theta-scheme (backward in time).
        The PDE is assumed to be:
            V_t + a(t, S) V_SS + b(t, S) V_S + c(t, S) V + f(t, S) = 0

        The implementation builds and solves a tridiagonal linear system at each time step.
        Preconditions:
        - The instance provides attributes: T, N, M, S_inf, a, b, c, f, theta.
        - The instance provides methods: apply_payoff(S, V) and apply_S0(t, V).
        - A tridiagonal solver `_thomas_solver(lower, diag, upper, rhs)` is available.
        """
        # Validate required attributes
        required_attrs = ("T", "N", "M", "S_inf", "a", "b", "c", "f", "theta")
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute '{attr}' in OptionSolver instance.")

        # Time and space discretization
        dt = float(self.T) / (self.N + 1)
        dS = float(self.S_inf) / (self.M + 1)

        # Coefficient multipliers
        alpha = 1.0 / (dS * dS)
        beta = 1.0 / dS
        gamma = 1.0 / dt

        theta = float(self.theta)
        one_minus_theta = 1.0 - theta

        # Mesh grids (kept for compatibility with coefficient functions that expect full mesh)
        t = np.linspace(0.0, float(self.T), self.N + 2)
        S = np.linspace(0.0, float(self.S_inf), self.M + 2)
        S_mesh, t_mesh = np.meshgrid(S, t, indexing="ij")

        # Helper to evaluate coefficient functions on mesh and normalize shapes
        def _eval_coef(fun, name="coef"):
            vals = fun(t_mesh, S_mesh)
            vals = np.asarray(vals, dtype=float)

            expected_shape = (self.M + 2, self.N + 2)
            if vals.shape != expected_shape:
                # Allow scalar or 1-D returns and broadcast appropriately
                if vals.shape == ():
                    vals = np.full(expected_shape, float(vals))
                elif vals.shape == (self.M + 2,):
                    vals = np.tile(vals.reshape(self.M + 2, 1), (1, self.N + 2))
                elif vals.shape == (self.N + 2,):
                    vals = np.tile(vals.reshape(1, self.N + 2), (self.M + 2, 1))
                else:
                    raise ValueError(f"{name} returned array with unexpected shape {vals.shape}. Expected {expected_shape}.")
            return vals

        # Evaluate PDE coefficient fields on mesh
        a_values = _eval_coef(self.a, "a")
        b_values = _eval_coef(self.b, "b")
        c_values = _eval_coef(self.c, "c")
        f_values = _eval_coef(self.f, "f")

        # Initialize solution array and apply boundary/final conditions
        V = np.zeros((self.M + 2, self.N + 2), dtype=float)
        self.apply_payoff(S, V)   # Final condition V(T, S)
        self.apply_S0(t, V)       # Boundary condition at S = 0 for all times

        # Preallocate arrays for tridiagonal matrix and RHS to avoid repeated allocations
        M = self.M
        A_diag = np.empty(M, dtype=float)
        A_lower = np.empty(M - 1, dtype=float)
        A_upper = np.empty(M - 1, dtype=float)

        B_diag = np.empty(M, dtype=float)
        B_lower = np.empty(M - 1, dtype=float)
        B_upper = np.empty(M - 1, dtype=float)

        rhs = np.empty(M, dtype=float)

        # Backward time-stepping loop: j indexes time levels (from N down to 0)
        for j in range(self.N, -1, -1):

            # Compute time-slice coefficients (vectors over S)
            eta_j = alpha * a_values[:, j]                    # length M+2
            phi_j = 2.0 * alpha * a_values[:, j] + beta * b_values[:, j] - c_values[:, j]
            psi_j = alpha * a_values[:, j] + beta * b_values[:, j]

            # Extract interior node coefficients for i = 1..M (size M)
            eta_i = eta_j[1:M+1]
            phi_i = phi_j[1:M+1]
            psi_i = psi_j[1:M+1]

            # Assemble tridiagonal matrix A coefficients (theta-weighted)
            # A_diag[i-1] corresponds to node i (i = 1..M)
            A_diag[:] = -gamma - theta * phi_i
            A_lower[:] = theta * eta_i[1:]    # subdiagonal entries for rows 2..M
            A_upper[:] = theta * psi_i[:-1]   # superdiagonal entries for rows 1..M-1

            # Assemble matrix B coefficients ((1-theta)-weighted)
            B_diag[:] = -gamma + one_minus_theta * phi_i
            B_lower[:] = -one_minus_theta * eta_i[1:]
            B_upper[:] = -one_minus_theta * psi_i[:-1]

            # Apply the special-last-row adjustments to match original indexing/BC logic
            # These override the last entries computed above
            aux1 = (eta_j[self.M - 1] - psi_j[self.M - 1])
            aux2 = (phi_j[self.M - 1] - 2.0 * psi_j[self.M - 1])
            A_lower[-1] = theta * aux1
            A_diag[-1] = -gamma - theta * aux2
            B_lower[-1] = -one_minus_theta * aux1
            B_diag[-1] = -gamma + one_minus_theta * aux2

            # Compute right-hand side: rhs = B @ V_{j+1} - F
            y = V[1:-1, j + 1]  # solution at next time level, interior nodes (size M)

            # Start with diagonal contribution
            rhs[:] = B_diag * y
            # Add off-diagonal contributions (vectorized)
            rhs[:-1] += B_upper * y[1:]   # superdiagonal * next node
            rhs[1:] += B_lower * y[:-1]   # subdiagonal * previous node

            # Add source term contribution: F = theta * f(t_j, S) + (1-theta) * f(t_{j+1}, S)
            F = (theta * f_values[1:-1, j] + one_minus_theta * f_values[1:-1, j + 1]).astype(float)
            rhs -= F

            # Apply any explicit constant adjustments
            rhs[0] += - theta * eta_j[1] * V[0, j] - (1 - theta) * eta_j[1] * V[0, j + 1]

            # Solve tridiagonal system A * x = rhs (Thomas algorithm)
            # The helper _thomas_solver(lower, diag, upper, rhs) must return length-M vector
            V[1:-1, j] = _thomas_solver(A_lower, A_diag, A_upper, rhs)

            # Far-field boundary at S = S_inf: linear extrapolation
            V[-1, j] = 2.0 * V[-2, j] - V[-3, j]

        # Persist solution and meshes on the instance for downstream use
        self.V = V
        self.t = t_mesh
        self.S = S_mesh



class CallCompoundOption(CompoundOption):
    """
    Class for pricing call compound options.
    """

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
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")

class BinaryCallCompoundOption(CompoundOption):
    """
    Class for pricing binary (cash-or-nothing) call compound options.
    """

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
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")

class BinaryPutCompoundOption(CompoundOption):
    """
    Class for pricing binary (cash-or-nothing) put compound options.
    """

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
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Utils.Parte6.DF.EuropeanOption import CallEuropean, PutEuropean, BinaryCallEuropean, BinaryPutEuropean

    T_C0 = 0.5
    T = 1

    call = CallEuropean(T = T)
    put = PutEuropean(T = T)
    binary_call = BinaryCallEuropean(T = T)
    binary_put = BinaryPutEuropean(T = T)

    # Call on call
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_call = CallCompoundOption(call, T=T_C0)
    call_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    call.plot_value_at_t(0, fig=fig, ax=ax, label="Call Option")
    ax.set_title("Compound Call Option on Call Option")

    # Call on put
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_put = CallCompoundOption(put, T=T_C0)
    call_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    put.plot_value_at_t(0, fig=fig, ax=ax, label="Put Option")
    ax.set_title("Compound Call Option on Put Option")

    # Call on binary call
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_binary_call = CallCompoundOption(binary_call, T=T_C0)
    call_binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call Option")
    ax.set_title("Compound Call Option on Binary Call Option")

    # Call on binary put
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    call_binary_put = CallCompoundOption(binary_put, T=T_C0)
    call_binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Call Option")
    binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put Option")
    ax.set_title("Compound Call Option on Binary Put Option")

    # Put on call
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_call = PutCompoundOption(call, T=T_C0)
    put_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    call.plot_value_at_t(0, fig=fig, ax=ax, label="Call Option")
    ax.set_title("Compound Put Option on Call Option")

    # Put on put
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_put = PutCompoundOption(put, T=T_C0)
    put_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    put.plot_value_at_t(0, fig=fig, ax=ax, label="Put Option")
    ax.set_title("Compound Put Option on Put Option")

    # Put on binary call
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_binary_call = PutCompoundOption(binary_call, T=T_C0)
    put_binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    binary_call.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Call Option")
    ax.set_title("Compound Put Option on Binary Call Option")

    # Put on binary put
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid()
    put_binary_put = PutCompoundOption(binary_put, T=T_C0)
    put_binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Compound Put Option")
    binary_put.plot_value_at_t(0, fig=fig, ax=ax, label="Binary Put Option")
    ax.set_title("Compound Put Option on Binary Put Option")

    

    plt.show()