import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _thomas_solver(lower, diag, upper, d):
    """
    Thomas algorithm for tridiagonal system.
    lower: length n-1 (subdiagonal)
    diag:  length n   (diagonal)
    upper: length n-1 (superdiagonal)
    d:     length n   (rhs)
    returns x length n
    """
    n = diag.size
    if n == 0:
        return np.array([])
    # Copy arrays to avoid modifying the original inputs
    a = diag.astype(float).copy()
    b = upper.astype(float).copy()
    c = lower.astype(float).copy()
    d = d.astype(float).copy()

    # Forward elimination: transforms the system into upper triangular form
    for i in range(1, n):
        if a[i-1] == 0:
            raise np.linalg.LinAlgError("Zero pivot in Thomas solver")
        m = c[i-1] / a[i-1]
        a[i] = a[i] - m * b[i-1]      # Update diagonal element
        d[i] = d[i] - m * d[i-1]      # Update right-hand side

    # Back substitution: solves for the unknowns starting from the last equation
    x = np.zeros(n, dtype=float)
    if a[-1] == 0:
        raise np.linalg.LinAlgError("Zero pivot in Thomas solver")
    x[-1] = d[-1] / a[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - b[i] * x[i+1]) / a[i]
    return x

def DF_EDP_parabolica(a, b, c, f,
                      func_cond: callable = None,
                      T: float = 1.0, N: int = 100, S_inf: float = 10.0, M: int = 100,
                      theta: float = 0.5):
    """
    Finite difference solver for a parabolic PDE of the form:
        V_t + a(t,S) V_SS + b(t,S) V_S + c(t,S) V + f(t,S) = 0

    Parameters:
        a, b, c, f: coefficient functions (callable)
        func_cond: function to set boundary and terminal conditions
        T: final time
        N: number of time steps
        S_inf: maximum value of S (spatial domain)
        M: number of spatial steps
        theta: theta parameter (0=explicit, 1=implicit, 0.5=Crank-Nicolson)

    Returns:
        S_mesh, t_mesh: grids for S and t
        V: solution matrix
    """
    # Step sizes
    Dt = T / (N + 1)
    DS = S_inf / (M + 1)

    alpha = 1.0 / (DS ** 2)
    beta = 1.0 / DS
    gamma = 1.0 / Dt

    # Create grids for time and space
    t = np.linspace(0.0, T, N + 2) 
    S = np.linspace(0.0, S_inf, M + 2)

    S_mesh, t_mesh = np.meshgrid(S, t, indexing='xy')
    t_mesh = t_mesh.T
    S_mesh = S_mesh.T

    def _eval_coef(fun, name="coef"):
        """
        Evaluate coefficient function on the mesh.
        Handles scalar, 1D, and 2D outputs.
        """
        try:
            vals = fun(t_mesh, S_mesh)
            vals = np.asarray(vals, dtype=float)
            if vals.shape != (M + 2, N + 2):
                if vals.shape == ():
                    vals = np.full((M + 2, N + 2), float(vals))
                elif vals.shape == (M + 2,):
                    vals = np.tile(vals.reshape(M + 2, 1), (1, N + 2))
                elif vals.shape == (N + 2,):
                    vals = np.tile(vals.reshape(1, N + 2), (M + 2, 1))
                else:
                    raise ValueError(f"{name} returned array with unexpected shape {vals.shape}")
        except Exception:
            try:
                s = float(fun(0.0, 0.0))
                vals = np.full((M + 2, N + 2), s, dtype=float)
            except Exception as e:
                raise ValueError(f"Could not evaluate coefficient {name}: {e}")
        return vals

    # Evaluate coefficients on the mesh
    a_values = _eval_coef(a, "a")
    b_values = _eval_coef(b, "b")
    c_values = _eval_coef(c, "c")
    f_values = _eval_coef(f, "f")

    # Initialize solution matrix
    V = np.zeros((M + 2, N + 2), dtype=float)

    # Apply boundary and terminal conditions
    V = func_cond(S, t, V)

    # Time-stepping loop (backward in time)
    for j in range(N, -1, -1):
        # Coefficients at time level j (for matrix A)
        eta_j = alpha * a_values[:, j]
        phi_j = 2.0 * alpha * a_values[:, j] + beta * b_values[:, j] - c_values[:, j]
        psi_j = alpha * a_values[:, j] + beta * b_values[:, j]

        # Coefficients at time level j+1 (for matrix B)
        eta_j1 = alpha * a_values[:, j + 1]
        phi_j1 = 2.0 * alpha * a_values[:, j + 1] + beta * b_values[:, j + 1] - c_values[:, j + 1]
        psi_j1 = alpha * a_values[:, j + 1] + beta * b_values[:, j + 1]

        # Build tridiagonal matrix A (for V_j)
        A_diag = np.empty(M, dtype=float)
        A_lower = np.empty(M - 1, dtype=float)  # subdiagonal
        A_upper = np.empty(M - 1, dtype=float)  # superdiagonal

        for k in range(M):
            i = k + 1
            A_diag[k] = -gamma - theta * phi_j[i]
            if k > 0:
                A_lower[k - 1] = theta * eta_j[i]
            if k < M - 1:
                A_upper[k] = theta * psi_j[i]

        # Build tridiagonal matrix B (for V_{j+1})
        B_lower = np.empty(M - 1, dtype=float)
        B_diag = np.empty(M, dtype=float)
        B_upper = np.empty(M - 1, dtype=float)
        for k in range(M):
            i = k + 1
            B_diag[k] = -(gamma - (1.0 - theta) * phi_j1[i])
            if k > 0:
                B_lower[k - 1] = -(1.0 - theta) * eta_j1[i]
            if k < M - 1:
                B_upper[k] = -(1.0 - theta) * psi_j1[i]

        # Weighted source term
        F = (theta * f_values[1:-1, j] + (1.0 - theta) * f_values[1:-1, j + 1]).astype(float)

        # Boundary contributions (first and last equations)
        C = np.zeros(M, dtype=float)
        C[0] = - (theta * eta_j[1]) * V[0, j] + (-(1.0 - theta) * eta_j1[1]) * V[0, j + 1]
        C[-1] = - (theta * psi_j[M]) * V[-1, j] + (-(1.0 - theta) * psi_j1[M]) * V[-1, j + 1]

        # Compute right-hand side: B @ V_{j+1} - F + C
        y = V[1:-1, j + 1]  # interior values at time j+1
        rhs = np.empty(M, dtype=float)
        for k in range(M):
            val = B_diag[k] * y[k]
            if k > 0:
                val += B_lower[k - 1] * y[k - 1]
            if k < M - 1:
                val += B_upper[k] * y[k + 1]
            rhs[k] = val

        rhs = rhs - F + C

        # Solve tridiagonal system: A x = rhs
        x = _thomas_solver(A_lower, A_diag, A_upper, rhs)

        # Update solution at time j
        V[1:-1, j] = x

    return S_mesh, t_mesh, V



# ---------------------------------------------------------------------------
# EUROPEAN OPTIONS
# ---------------------------------------------------------------------------

class EuropeanOption:
    def __init__(self, r=0.05, K=20, D=0, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5, **kwargs):
        self.r = r
        self.K = K
        self.D = D
        self.T = T
        self.S_inf = S_inf
        self.EDE = EDE
        self.N = N
        self.M = M
        self.theta = theta
        if EDE == 'lognormal':
            self.sigma = kwargs.get('sigma', 0.2)

    def solve(self):
        if self.EDE == 'lognormal':
            self.S, self.t, self.V = DF_EDP_parabolica(
                a = lambda t, S: self.sigma**2 * S**2 / 2,
                b = lambda t, S: (self.r-self.D) * S,
                c = lambda t, S: -self.r,
                f = lambda t, S: 0,
                func_cond = self.set_conditions,
                T = self.T,
                S_inf = self.S_inf,
                N = self.N,
                M = self.M,
                theta = self.theta
            )

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.S, self.t, self.V, cmap='viridis')
        ax.set_xlabel('S')
        ax.set_ylabel('t')
        ax.set_zlabel('Value')
        
        return fig, ax

    def set_conditions(self):
        pass


class CallEuropean(EuropeanOption):

    def plot(self):
        fig, ax = super().plot()
        ax.set_title(f"European Call Option (K={self.K})")
        return fig, ax

    def set_conditions(self, S, t, V):
        #Payoff
        payoff = lambda S: np.maximum(S - self.K, 0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        #Condiciones de frontera
        func_S_inf = lambda t: self.S_inf * np.exp(-self.D*(self.T-t)) - self.K * np.exp(-self.r * (self.T - t))
        func_S0 = lambda t: 0
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V

class PutEuropean(EuropeanOption):

    def plot(self):
        fig, ax = super().plot()
        ax.set_title(f"European Put Option (K={self.K})")
        return fig, ax

    def set_conditions(self, S, t, V):
        # Payoff
        payoff = lambda S: np.maximum(self.K - S, 0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Condiciones de frontera
        func_S_inf = lambda t: 0
        func_S0 = lambda t: self.K * np.exp(-self.r * (self.T - t))
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V
    
class BinaryCallEuropean(EuropeanOption):
    def __init__(self, r=0.05, K=20, D=0, P=1, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5, **kwargs):
        super().__init__(r=r, K=K, D=D, T=T, S_inf=S_inf, EDE=EDE, N=N, M=M, theta=theta, **kwargs)
        self.P = P

    def plot(self):
        fig, ax = super().plot()
        ax.set_title(f"Binary Call Option (K={self.K}, P={self.P})")
        return fig, ax
    
    def set_conditions(self, S, t, V):
        # Payoff binario
        payoff = lambda S: np.where(S > self.K, self.P, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Condiciones de frontera
        func_S_inf = lambda t: self.P * np.exp(-self.r * (self.T - t))
        func_S0 = lambda t: 0.0
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V
    
class BinaryPutEuropean(EuropeanOption):
    def __init__(self, r=0.05, K=20, D=0, P=1, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5, **kwargs):
        super().__init__(r=r, K=K, D=D, T=T, S_inf=S_inf, EDE=EDE, N=N, M=M, theta=theta, **kwargs)
        self.P = P

    def plot(self):
        fig, ax = super().plot()
        ax.set_title(f"Binary Put Option (K={self.K}, P={self.P})")
        return fig, ax
    
    def set_conditions(self, S, t, V):
        # Payoff binario
        payoff = lambda S: np.where(S < self.K, self.P, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Condiciones de frontera
        func_S_inf = lambda t: 0.0
        func_S0 = lambda t: self.P * np.exp(-self.r * (self.T - t))
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V
    

calleur = CallEuropean()
calleur.solve()
fig, ax = calleur.plot()

puteur = PutEuropean()
puteur.solve()
fig, ax = puteur.plot()

bincall = BinaryCallEuropean()
bincall.solve()
fig, ax = bincall.plot()

binput = BinaryPutEuropean()
binput.solve()
fig, ax = binput.plot()

plt.show()


