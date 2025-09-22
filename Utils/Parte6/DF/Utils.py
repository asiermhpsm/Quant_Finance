import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# Auxiliary functions
# ---------------------------------------------------------------------------

def _thomas_solver(lower, diag, upper, d):
    """
    Thomas algorithm for a tridiagonal system.

    Parameters:
        lower: Subdiagonal (length n-1).
        diag:  Diagonal (length n).
        upper: Superdiagonal (length n-1).
        d:     Right-hand side (length n).

    Returns:
        x: Solution array (length n).
    """
    n = diag.size
    if n == 0:
        return np.array([])

    # Copy inputs to avoid in-place modifications.
    a = diag.astype(float).copy()
    b = upper.astype(float).copy()
    c = lower.astype(float).copy()
    d = d.astype(float).copy()

    # Forward elimination: transform the matrix to upper triangular form.
    for i in range(1, n):
        if a[i - 1] == 0:
            raise np.linalg.LinAlgError("Zero pivot in Thomas solver.")
        m = c[i - 1] / a[i - 1]
        a[i] = a[i] - m * b[i - 1]      # Update diagonal element.
        d[i] = d[i] - m * d[i - 1]      # Update right-hand side.

    # Back substitution: solve from the last equation upwards.
    x = np.zeros(n, dtype=float)
    if a[-1] == 0:
        raise np.linalg.LinAlgError("Zero pivot in Thomas solver.")
    x[-1] = d[-1] / a[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - b[i] * x[i + 1]) / a[i]
    return x

def DF_EDP_parabolica(a, b, c, f,
                      func_cond: callable = None,
                      T: float = 1.0, N: int = 100, S_inf: float = 10.0, M: int = 100,
                      theta: float = 0.5,
                      dividends: list | None = None,
                      early_exercise_payoff = None):
    """
    Finite-difference solver for a parabolic PDE of the form:
        V_t + a(t, S) V_SS + b(t, S) V_S + c(t, S) V + f(t, S) = 0.

    Parameters:
        a, b, c, f: Coefficient functions (callable).
        func_cond: Function to set boundary and terminal conditions.
        T: Final time.
        N: Number of time steps.
        S_inf: Maximum S (spatial domain).
        M: Number of spatial steps.
        theta: Theta parameter (0 = explicit, 1 = implicit, 0.5 = Crank–Nicolson).
        dividends: List of (t_i, D_i) for discrete dividend jumps at times t_i.
                   European:  V(S, t_i^-) = V(S - D_i, t_i^+).
                   American (if early_exercise_payoff is provided):
                              V(S, t_i^-) = max(V(S - D_i, t_i^+), Φ(S)).
        early_exercise_payoff: Φ(S) callable. If not None, apply the American exercise rule on dividend dates.

    Returns:
        S_mesh, t_mesh: Grids for S and t.
        V: Solution matrix.
    """
    # Step sizes.
    Dt = T / (N + 1)
    DS = S_inf / (M + 1)

    alpha = 1.0 / (DS ** 2)
    beta = 1.0 / DS
    gamma = 1.0 / Dt

    # Create time and space grids.
    t = np.linspace(0.0, T, N + 2)
    S = np.linspace(0.0, S_inf, M + 2)

    S_mesh, t_mesh = np.meshgrid(S, t, indexing='xy')
    t_mesh = t_mesh.T
    S_mesh = S_mesh.T

    # Map dividend times to the nearest time index on the mesh.
    div_map = {}
    if dividends:
        for (ti, Di) in dividends:
            idx = int(round(ti / Dt))
            idx = max(0, min(N + 1, idx))
            div_map[idx] = div_map.get(idx, 0.0) + float(Di)

    def _eval_coef(fun, name="coef"):
        """
        Evaluate a coefficient function on the mesh.

        Accepts scalar, 1D, or 2D outputs and broadcasts to shape (M + 2, N + 2).
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
                    raise ValueError(f"{name} returned array with unexpected shape {vals.shape}.")
        except Exception:
            try:
                s = float(fun(0.0, 0.0))
                vals = np.full((M + 2, N + 2), s, dtype=float)
            except Exception as e:
                raise ValueError(f"Could not evaluate coefficient {name}: {e}")
        return vals

    # Evaluate coefficients on the mesh.
    a_values = _eval_coef(a, "a")
    b_values = _eval_coef(b, "b")
    c_values = _eval_coef(c, "c")
    f_values = _eval_coef(f, "f")

    # Initialize solution matrix.
    V = np.zeros((M + 2, N + 2), dtype=float)

    # Apply boundary and terminal conditions.
    V = func_cond(S, t, V)

    # Helper to evaluate Φ(S) (accepts Φ(S) or Φ(S,t)).
    def _eval_payoff_phi(S_vec, t_val):
        if early_exercise_payoff is None:
            return None
        try:
            phi = early_exercise_payoff(S_vec, t_val)
        except TypeError:
            phi = early_exercise_payoff(S_vec)
        return np.asarray(phi, dtype=float)

    # Time-stepping loop (backward in time).
    for j in range(N, -1, -1):
        # Coefficients at time level j (matrix A).
        eta_j = alpha * a_values[:, j]
        phi_j = 2.0 * alpha * a_values[:, j] + beta * b_values[:, j] - c_values[:, j]
        psi_j = alpha * a_values[:, j] + beta * b_values[:, j]

        # Coefficients at time level j + 1 (matrix B).
        eta_j1 = alpha * a_values[:, j + 1]
        phi_j1 = 2.0 * alpha * a_values[:, j + 1] + beta * b_values[:, j + 1] - c_values[:, j + 1]
        psi_j1 = alpha * a_values[:, j + 1] + beta * b_values[:, j + 1]

        # Build tridiagonal matrix A (for V_j).
        A_diag = np.empty(M, dtype=float)
        A_lower = np.empty(M - 1, dtype=float)  # Subdiagonal.
        A_upper = np.empty(M - 1, dtype=float)  # Superdiagonal.

        for k in range(M):
            i = k + 1
            A_diag[k] = -gamma - theta * phi_j[i]
            if k > 0:
                A_lower[k - 1] = theta * eta_j[i]
            if k < M - 1:
                A_upper[k] = theta * psi_j[i]

        # Build tridiagonal matrix B (for V_{j+1}).
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

        # Weighted source term.
        F = (theta * f_values[1:-1, j] + (1.0 - theta) * f_values[1:-1, j + 1]).astype(float)

        # Boundary contributions (first and last equations).
        C = np.zeros(M, dtype=float)
        C[0] = -(theta * eta_j[1]) * V[0, j] + (-(1.0 - theta) * eta_j1[1]) * V[0, j + 1]
        C[-1] = -(theta * psi_j[M]) * V[-1, j] + (-(1.0 - theta) * psi_j1[M]) * V[-1, j + 1]

        # Compute right-hand side: B @ V_{j+1} - F + C.
        y = V[1:-1, j + 1]  # Interior values at time j + 1.
        rhs = np.empty(M, dtype=float)
        for k in range(M):
            val = B_diag[k] * y[k]
            if k > 0:
                val += B_lower[k - 1] * y[k - 1]
            if k < M - 1:
                val += B_upper[k] * y[k + 1]
            rhs[k] = val

        rhs = rhs - F + C

        # Solve tridiagonal system A x = rhs.
        x = _thomas_solver(A_lower, A_diag, A_upper, rhs)
        V[1:-1, j] = x

        # Dividend jump at t_j.
        if j in div_map:
            Dj = div_map[j]
            V_plus = V[:, j].copy()  # V(S, t_j^+)
            S_shift = S - Dj
            V_jump = np.interp(S_shift, S, V_plus, left=V_plus[0], right=V_plus[-1])  # V(S - D_j, t_j^+)
            if early_exercise_payoff is not None:
                Phi = _eval_payoff_phi(S, t[j])  # Φ(S)
                if Phi.shape != V_jump.shape:
                    raise ValueError("early_exercise_payoff must return a vector of size M+2.")
                V[:, j] = np.maximum(V_jump, Phi)
            else:
                V[:, j] = V_jump

    return S_mesh, t_mesh, V


def plot_surface(S, t, V, xlabel='S', ylabel='t', zlabel='Value', title=None, fig=None, ax=None, **surface_kwargs):
    """
    Plot a 3D surface of V(S, t). If fig/ax are provided, draw on them.

    Parameters:
        S, t: Meshgrid arrays for S and t.
        V: Value array.
        xlabel, ylabel, zlabel: Axis labels.
        title: Plot title.
        fig, ax: Optional matplotlib Figure and 3D Axes to draw on.
        surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    created_ax = False
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        created_ax = True
    else:
        # Ensure it's a 3D axis
        if not hasattr(ax, "plot_surface"):
            raise ValueError('The provided axis is not 3D. Create one with subplot_kw={"projection": "3d"}.')
        if fig is None:
            fig = ax.figure

    # Default colormap if none given
    if "cmap" not in surface_kwargs:
        surface_kwargs["cmap"] = "viridis"

    ax.plot_surface(S, t, V, **surface_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    # Adjust view only if the axis was created here
    if created_ax:
        ax.view_init(elev=30, azim=-135)
    return fig, ax

def plot_func(x, y, fig=None, ax=None, xlabel='x', ylabel='f(x)', **plot_kwargs):
    """
    Plots a function in 2D.
    """
    created_ax = False
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        created_ax = True
    else:
        if fig is None:
            fig = ax.figure

    ax.plot(x, y, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if created_ax:
        ax.grid()
    return fig, ax

# ---------------------------------------------------------------------------
# European options
# ---------------------------------------------------------------------------

class EuropeanOption:
    """
    Base class for European options solved via finite differences.
    """
    def __init__(self, r=0.05, K=20, D=0, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5,
                 dividends: list | None = None, **kwargs):
        self.r = r
        self.K = K
        self.D = D  # Continuous yield (if used).
        self.T = T
        self.S_inf = S_inf
        self.EDE = EDE
        self.N = N
        self.M = M
        self.theta = theta
        self.dividends = dividends or []

        # Model-specific parameters
        if self.EDE == 'lognormal':
            self.sigma = kwargs.get('sigma', 0.2)

            # PDE coefficients stored as attributes
            # V_t + a(t,S) V_SS + b(t,S) V_S + c(t,S) V + f(t,S) = 0
            a = lambda t, S: 0.5 * (self.sigma ** 2) * S**2
            b = lambda t, S: (self.r - self.D) * S
            c = lambda t, S: -self.r
            f = lambda t, S: 0.0

            self.a = a
            self.b = b
            self.c = c
            self.f = f
        else:
            raise NotImplementedError(f"EDE '{self.EDE}' not implemented.")

    def set_conditions(self, V):
        """
        Placeholder to be implemented in derived classes.
        """
        return V

    def get_solution(self):
        """
        Returns S, t, V without plotting.
        """
        if not hasattr(self, "V"):
            self.solve()
        return self.S, self.t, self.V


    def solve(self):
        """
        Solve the PDE using the configured parameters and store S, t, and V.
        """
        self.S, self.t, self.V = DF_EDP_parabolica(
            a=self.a,
            b=self.b,
            c=self.c,
            f=self.f,
            func_cond=self.set_conditions,
            T=self.T,
            S_inf=self.S_inf,
            N=self.N,
            M=self.M,
            theta=self.theta,
            dividends=self.dividends
        )

    def plot(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the surface V(S, t). If fig/ax are provided, draw on them.
        """
        if not hasattr(self, "V"):
            self.solve()
        return plot_surface(self.S, self.t, self.V, xlabel='S', ylabel='t', zlabel='Value', fig=fig, ax=ax, **surface_kwargs)


    def get_value_at_S(self, S0: float):
        """Returns (t, V(S0, t)) as 1D arrays."""
        if not hasattr(self, "V"):
            self.solve()
        DS = self.S_inf / (self.M + 1)
        idx = int(round(S0 / DS))
        return self.t[0, :], self.V[idx, :]

    def plot_value_at_S(self, S0: float, fig=None, ax=None, **plot_kwargs):
        """
        Plot V(S0, t) as a function of t.
        """
        t, V_S0 = self.get_value_at_S(S0)
        return plot_func(t, V_S0, xlabel='t', ylabel=f'V(S={S0}, t)', fig=fig, ax=ax, **plot_kwargs)
    
    def get_value_at_t(self, t0: float):
        """Returns (S, V(S, t0)) as 1D arrays."""
        if not hasattr(self, "V"):
            self.solve()
        Dt = self.T / (self.N + 1)
        j = int(round(t0 / Dt))
        return self.S[:, 0], self.V[:, j]
    
    def plot_value_at_t(self, t0: float, fig=None, ax=None, **plot_kwargs):
        """
        Plot V(S, t0) as a function of S.
        """
        S, V_t0 = self.get_value_at_t(t0)
        return plot_func(S, V_t0, xlabel='S', ylabel=f'V(S, t={t0})', fig=fig, ax=ax, **plot_kwargs)


    def get_delta(self):
        """
        Compute the option delta using the theta-weighted finite difference scheme:
            delta ≈ θ * (V_{j}^{i+1} - V_{j}^{i}) / ΔS + (1 - θ) * (V_{j+1}^{i+1} - V_{j+1}^{i}) / ΔS
        Returns:
            delta: Array of delta values with shape (M + 2, N + 2)
        """
        # Ensure the solution is available
        if not hasattr(self, "V"):
            self.solve()

        # Get grid sizes
        DS = self.S_inf / (self.M + 1)

        # Prepare delta array
        delta = np.full_like(self.V, np.nan)

        # Loop over all grid points except last in S and t
        for j in range(self.V.shape[1] - 1):
            for i in range(self.V.shape[0] - 1):
                # Finite difference as per theta-scheme
                d1 = (self.V[i + 1, j] - self.V[i, j]) / DS
                d2 = (self.V[i + 1, j + 1] - self.V[i, j + 1]) / DS
                delta[i, j] = self.theta * d1 + (1 - self.theta) * d2
        
        self.delta = delta
        return delta
    
    def plot_delta(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the delta surface.
        """
        if not hasattr(self, "delta"):
            self.get_delta()
        return plot_surface(self.S, self.t, self.delta, xlabel='S', ylabel='t', zlabel='Delta', fig=fig, ax=ax, **surface_kwargs)


    def get_gamma(self):
        """
        Compute the option gamma using the theta-weighted finite difference scheme:
            gamma ≈ θ * (V_{j}^{i+1} - 2V_{j}^{i} + V_{j}^{i-1}) / ΔS^2
                  + (1 - θ) * (V_{j+1}^{i+1} - 2V_{j+1}^{i} + V_{j+1}^{i-1}) / ΔS^2
        Returns:
            gamma: Array of gamma values with shape (M + 2, N + 2)
        """
        if not hasattr(self, "V"):
            self.solve()

        DS = self.S_inf / (self.M + 1)
        gamma = np.full_like(self.V, np.nan)

        for j in range(self.V.shape[1] - 1):
            for i in range(1, self.V.shape[0] - 1):
                g1 = (self.V[i + 1, j] - 2 * self.V[i, j] + self.V[i - 1, j]) / (DS ** 2)
                g2 = (self.V[i + 1, j + 1] - 2 * self.V[i, j + 1] + self.V[i - 1, j + 1]) / (DS ** 2)
                gamma[i, j] = self.theta * g1 + (1 - self.theta) * g2

        self.gamma = gamma
        return gamma

    def plot_gamma(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the gamma surface.
        """
        if not hasattr(self, "gamma"):
            self.get_gamma()
        return plot_surface(self.S, self.t, self.gamma, xlabel='S', ylabel='t', zlabel='Gamma', fig=fig, ax=ax, **surface_kwargs)


    def get_theta(self):
        """
        Compute the option theta using the finite difference scheme:
            theta ≈ (V_{j+1}^{i} - V_{j}^{i}) / Δt
        Returns:
            theta: Array of theta values with shape (M + 2, N + 2)
        """
        if not hasattr(self, "V"):
            self.solve()

        Dt = self.T / (self.N + 1)
        theta_arr = np.full_like(self.V, np.nan)

        for j in range(self.V.shape[1] - 1):
            for i in range(self.V.shape[0]):
                theta_arr[i, j] = (self.V[i, j + 1] - self.V[i, j]) / Dt

        self.theta_arr = theta_arr
        return theta_arr

    def plot_theta(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the theta surface.
        """
        if not hasattr(self, "theta_arr"):
            self.get_theta()
        return plot_surface(self.S, self.t, self.theta_arr, xlabel='S', ylabel='t', zlabel='Theta', fig=fig, ax=ax, **surface_kwargs)


    def get_speed(self):
        """
        Compute the option speed (third derivative with respect to S) using the theta-weighted finite difference scheme:
            speed ≈ θ * (-V_{j}^{i-2} + 2V_{j}^{i-1} - 2V_{j}^{i+1} + V_{j}^{i+2}) / (8 ΔS^3)
                    + (1 - θ) * (-V_{j+1}^{i-2} + 2V_{j+1}^{i-1} - 2V_{j+1}^{i+1} + V_{j+1}^{i+2}) / (8 ΔS^3)
        Returns:
            speed: Array of speed values with shape (M + 2, N + 2)
        """
        if not hasattr(self, "V"):
            self.solve()

        DS = self.S_inf / (self.M + 1)
        speed = np.full_like(self.V, np.nan)

        for j in range(self.V.shape[1] - 1):
            for i in range(2, self.V.shape[0] - 2):
                s1 = (-self.V[i - 2, j] + 2 * self.V[i - 1, j] - 2 * self.V[i + 1, j] + self.V[i + 2, j]) / (8 * DS ** 3)
                s2 = (-self.V[i - 2, j + 1] + 2 * self.V[i - 1, j + 1] - 2 * self.V[i + 1, j + 1] + self.V[i + 2, j + 1]) / (8 * DS ** 3)
                speed[i, j] = self.theta * s1 + (1 - self.theta) * s2

        self.speed = speed
        return speed

    def plot_speed(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the speed surface.
        """
        if not hasattr(self, "speed"):
            self.get_speed()
        return plot_surface(self.S, self.t, self.speed, xlabel='S', ylabel='t', zlabel='Speed', fig=fig, ax=ax, **surface_kwargs)


    def get_vega(self, dsigma=1e-6):
        """
        Approximate the option vega (∂V/∂σ) using central finite differences.
        Returns:
            vega: Array of vega values with shape (M + 2, N + 2)
        """
        if self.EDE != 'lognormal':
            raise NotImplementedError("Vega computation only implemented for lognormal model.")
        
        if not hasattr(self, "V"):
            self.solve()

        sigma_orig = getattr(self, "sigma", 0.2)

        def _solve_with_sigma(sig):
            self.sigma = sig
            _, _, V_tmp = DF_EDP_parabolica(
                a=self.a, b=self.b, c=self.c, f=self.f,
                func_cond=self.set_conditions,
                T=self.T, S_inf=self.S_inf,
                N=self.N, M=self.M, theta=self.theta,
                dividends=self.dividends
            )
            return V_tmp

        V_p = _solve_with_sigma(sigma_orig + dsigma)
        V_m = _solve_with_sigma(sigma_orig - dsigma)
        self.sigma = sigma_orig

        vega = (V_p - V_m) / (2 * dsigma)
        self.vega = vega
        return vega

    def plot_vega(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the vega surface.
        """
        if not hasattr(self, "vega"):
            self.get_vega()
        return plot_surface(self.S, self.t, self.vega, xlabel='S', ylabel='t', zlabel='Vega', fig=fig, ax=ax, **surface_kwargs)



    def get_rho_r(self, dr=1e-6):
        """
        Approximate the option rho (∂V/∂r) using central finite differences.
        Returns:
            rho_r: Array of rho values with shape (M + 2, N + 2)
        """
        if not hasattr(self, "V"):
            self.solve()

        r_orig = self.r

        def _solve_with_r(rval):
            self.r = rval
            _, _, V_tmp = DF_EDP_parabolica(
                a=self.a, b=self.b, c=self.c, f=self.f,
                func_cond=self.set_conditions,
                T=self.T, S_inf=self.S_inf,
                N=self.N, M=self.M, theta=self.theta,
                dividends=self.dividends
            )
            return V_tmp

        V_p = _solve_with_r(r_orig + dr)
        V_m = _solve_with_r(r_orig - dr)
        self.r = r_orig

        rho_r = (V_p - V_m) / (2 * dr)
        self.rho_r = rho_r
        return rho_r

    def plot_rho_r(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the rho (with respect to r) surface.
        """
        if not hasattr(self, "rho_r"):
            self.get_rho_r()
        return plot_surface(self.S, self.t, self.rho_r, xlabel='S', ylabel='t', zlabel='Rho (r)', fig=fig, ax=ax, **surface_kwargs)


    def get_rho_D(self, dD=1e-6):
        """
        Approximate the option rho_D (∂V/∂D) using central finite differences.
        Returns:
            rho_D: Array of rho_D values with shape (M + 2, N + 2)
        """
        if not hasattr(self, "V"):
            self.solve()

        D_orig = self.D

        def _solve_with_D(Dval):
            self.D = Dval
            _, _, V_tmp = DF_EDP_parabolica(
                a=self.a, b=self.b, c=self.c, f=self.f,
                func_cond=self.set_conditions,
                T=self.T, S_inf=self.S_inf,
                N=self.N, M=self.M, theta=self.theta,
                dividends=self.dividends
            )
            return V_tmp

        V_p = _solve_with_D(D_orig + dD)
        V_m = _solve_with_D(D_orig - dD)
        self.D = D_orig

        rho_D = (V_p - V_m) / (2 * dD)
        self.rho_D = rho_D
        return rho_D

    def plot_rho_D(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the rho (with respect to D) surface.
        """
        if not hasattr(self, "rho_D"):
            self.get_rho_D()
        return plot_surface(self.S, self.t, self.rho_D, xlabel='S', ylabel='t', zlabel='Rho (D)', fig=fig, ax=ax, **surface_kwargs)


    def plot_all(self):
        """
        Plot the solution and all Greeks.
        """
        figs = []
        axs = []

        fig, ax = self.plot()
        figs.append(fig)
        axs.append(ax)

        fig_delta, ax_delta = self.plot_delta()
        figs.append(fig_delta)
        axs.append(ax_delta)

        fig_gamma, ax_gamma = self.plot_gamma()
        figs.append(fig_gamma)
        axs.append(ax_gamma)

        fig_theta, ax_theta = self.plot_theta()
        figs.append(fig_theta)
        axs.append(ax_theta)

        fig_speed, ax_speed = self.plot_speed()
        figs.append(fig_speed)
        axs.append(ax_speed)

        fig_vega, ax_vega = self.plot_vega()
        figs.append(fig_vega)
        axs.append(ax_vega)

        fig_rho_r, ax_rho_r = self.plot_rho_r()
        figs.append(fig_rho_r)
        axs.append(ax_rho_r)

        fig_rho_D, ax_rho_D = self.plot_rho_D()
        figs.append(fig_rho_D)
        axs.append(ax_rho_D)

        return figs, axs



class CallEuropean(EuropeanOption):
    """
    European call option with optional discrete dividends.
    """
    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"European Call Option (K={self.K})")
        return fig, ax

    def set_conditions(self, S, t, V):
        # Terminal payoff.
        payoff = lambda S: np.maximum(S - self.K, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Boundary conditions.
        func_S_inf = lambda tt: self.S_inf * np.exp(-self.D * (self.T - tt)) - self.K * np.exp(-self.r * (self.T - tt))
        func_S0 = lambda tt: 0.0
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V


class PutEuropean(EuropeanOption):
    """
    European put option with optional discrete dividends.
    """
    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"European Put Option (K={self.K})")
        return fig, ax

    def set_conditions(self, S, t, V):
        # Terminal payoff.
        payoff = lambda S: np.maximum(self.K - S, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Boundary conditions.
        func_S_inf = lambda tt: 0.0
        func_S0 = lambda tt: self.K * np.exp(-self.r * (self.T - tt))
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V


class BinaryCallEuropean(EuropeanOption):
    """
    European binary call option with optional discrete dividends.
    """
    def __init__(self, r=0.05, K=20, D=0, P=1, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5,
                 dividends: list | None = None, **kwargs):
        # Pass discrete dividends to the base class; the PDE solver will handle jumps.
        super().__init__(r=r, K=K, D=D, T=T, S_inf=S_inf, EDE=EDE, N=N, M=M, theta=theta,
                         dividends=dividends, **kwargs)
        self.P = P

    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"Binary Call Option (K={self.K}, P={self.P})")
        return fig, ax

    def set_conditions(self, S, t, V):
        # Terminal binary payoff.
        payoff = lambda S: np.where(S > self.K, self.P, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Boundary conditions.
        func_S_inf = lambda tt: self.P * np.exp(-self.r * (self.T - tt))
        func_S0 = lambda tt: 0.0
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V


class BinaryPutEuropean(EuropeanOption):
    """
    European binary put option with optional discrete dividends.
    """
    def __init__(self, r=0.05, K=20, D=0, P=1, T=1, S_inf=80, EDE='lognormal', N=200, M=200, theta=0.5,
                 dividends: list | None = None, **kwargs):
        # Pass discrete dividends to the base class; the PDE solver will handle jumps.
        super().__init__(r=r, K=K, D=D, T=T, S_inf=S_inf, EDE=EDE, N=N, M=M, theta=theta,
                         dividends=dividends, **kwargs)
        self.P = P

    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"Binary Put Option (K={self.K}, P={self.P})")
        return fig, ax

    def set_conditions(self, S, t, V):
        # Terminal binary payoff.
        payoff = lambda S: np.where(S < self.K, self.P, 0.0)
        V[:, -1] = np.asarray(payoff(S), dtype=float)

        # Boundary conditions.
        func_S_inf = lambda tt: 0.0
        func_S0 = lambda tt: self.P * np.exp(-self.r * (self.T - tt))
        V[0, :] = np.asarray(func_S0(t), dtype=float)
        V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

        return V


# ---------------------------------------------------------------------------
# American options
# ---------------------------------------------------------------------------

class CallAmerican(CallEuropean):
    def solve(self):
        super().solve()
        payoff = np.maximum(self.S - self.K, 0.0)
        self.V = np.maximum(self.V, payoff)
        
    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"American Call Option (K={self.K})")
        return fig, ax
    

class PutAmerican(PutEuropean):
    def solve(self):
        super().solve()
        payoff = np.maximum(self.K - self.S, 0.0)
        self.V = np.maximum(self.V, payoff)

    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"American Put Option (K={self.K})")
        return fig, ax


class BinaryCallAmerican(BinaryCallEuropean):
    def solve(self):
        super().solve()
        payoff = np.where(self.S > self.K, self.P, 0.0)
        self.V = np.maximum(self.V, payoff)

    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"American Binary Call Option (K={self.K}, P={self.P})")
        return fig, ax


class BinaryPutAmerican(BinaryPutEuropean):
    def solve(self):
        super().solve()
        payoff = np.where(self.S < self.K, self.P, 0.0)
        self.V = np.maximum(self.V, payoff)

    def plot(self, fig=None, ax=None, **surface_kwargs):
        fig, ax = super().plot(fig=fig, ax=ax, **surface_kwargs)
        ax.set_title(f"American Binary Put Option (K={self.K}, P={self.P})")
        return fig, ax





# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

# Example usage with discrete dividends for all options.
#divs = [(0.25, 2), (0.5, 5), (0.75, 10)]
divs = None

eur = CallEuropean(dividends=divs)
amer = CallAmerican(dividends=divs)

# Pintar ambas superficies en el mismo gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

eur.plot(fig=fig, ax=ax, alpha=0.8, cmap='viridis')
amer.plot(fig=fig, ax=ax, alpha=0.5, cmap='plasma')


fig2, ax2 = plt.subplots(figsize=(10, 7))
eur.plot_value_at_S(30, fig=fig2, ax=ax2, label='European Put', color='blue')
amer.plot_value_at_S(30, fig=fig2, ax=ax2, label='American Put', color='red')

fig3, ax3 = plt.subplots(figsize=(10, 7))
eur.plot_value_at_t(0, fig=fig3, ax=ax3, label='European Put', color='blue')
amer.plot_value_at_t(0, fig=fig3, ax=ax3, label='American Put', color='red')



plt.show()


