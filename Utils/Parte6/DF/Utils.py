import numpy as np
import warnings
import matplotlib.pyplot as plt



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

def plot_func(x, y, fig=None, ax=None, xlabel='x', ylabel='f(x)', title=None, label=None, **plot_kwargs):
    """
    Plots a function in 2D.
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure

    # Pass label to ax.plot if provided
    if label is not None:
        plot_kwargs['label'] = label
    ax.plot(x, y, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if title:
        ax.set_title(title)
    # Show legend if label is provided
    if label is not None:
        ax.legend()
    return fig, ax


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

def solve_PDE(a : callable, b: callable, c: callable, f: callable, F: callable, 
              S_min: float, S_max: float, T: float, 
              known_boundaries: str = 'None', 
              N: int = 500, M: int = 500, theta: float = 0.5,
              **kwargs
              ):
    """
    Solve the option pricing PDE using a finite-difference theta-scheme (backward in time).
        The PDE is assumed to be:
            V_t + a(t, S) V_SS + b(t, S) V_S + c(t, S) V + f(t, S) = 0
    Parameters:
        a, b, c, f (callable): Coefficient functions of the PDE.
        F (callable): final condition function at maturity T, i.e. V(T, S)=F(S).
        S_min, S_max (float): Spatial domain boundaries.
        T (float): Maturity time.
        known_boundaries (str): Known boundaries:
            - 'Both' : V(t, S_min) and V(t, S_max) known for all t.
            - 'S0'   : V(t, S_min) known for all t, V(t, S_max) unknown.
            - 'Smax' : V(t, S_max) known for all t, V(t, S_min) unknown.
            - 'None'  : Both boundaries unknown.
        N (int): Number of time steps.
        M (int): Number of spatial steps.
        theta (float): Theta parameter for the scheme.
        kwargs: Additional parameters for boundary conditions functions:
            - If known_boundaries is 'Both', provide S0_func(t) and Smax_func(t).
            - If known_boundaries is 'S0', provide S0_func(t).
            - If known_boundaries is 'Smax', provide Smax_func(t).
    Returns:
        S_grid, t_grid: Meshgrid arrays for S and t.
        V: 2D array of option values at each (t, S) grid point.
    """
    # Time and space discretization
    dt = float(T) / (N + 1)
    dS = float(S_max - S_min) / (M + 1)

    # Coefficient multipliers
    alpha = 1.0 / (dS * dS)
    beta = 1.0 / dS
    gamma = 1.0 / dt

    theta = float(theta)
    one_minus_theta = 1.0 - theta

    # Mesh grids (kept for compatibility with coefficient functions that expect full mesh)
    t = np.linspace(0.0, T, N + 2)
    S = np.linspace(0.0, S_max, M + 2)
    S_mesh, t_mesh = np.meshgrid(S, t, indexing="ij")

    # Helper to evaluate coefficient functions on mesh and normalize shapes
    def _eval_coef(fun, name="coef"):
        vals = fun(t_mesh, S_mesh)
        vals = np.asarray(vals, dtype=float)

        expected_shape = (M + 2, N + 2)
        if vals.shape != expected_shape:
            # Allow scalar or 1-D returns and broadcast appropriately
            if vals.shape == ():
                vals = np.full(expected_shape, float(vals))
            elif vals.shape == (M + 2,):
                vals = np.tile(vals.reshape(M + 2, 1), (1, N + 2))
            elif vals.shape == (N + 2,):
                vals = np.tile(vals.reshape(1, N + 2), (M + 2, 1))
            else:
                raise ValueError(f"{name} returned array with unexpected shape {vals.shape}. Expected {expected_shape}.")
        return vals

    # Evaluate PDE coefficient fields on mesh
    a_values = _eval_coef(a, "a")
    b_values = _eval_coef(b, "b")
    c_values = _eval_coef(c, "c")
    f_values = _eval_coef(f, "f")

    # Check that a(t, 0) and b(t, 0) are zero for all t
    if known_boundaries in ['Smax', 'None']:
        if not np.allclose(a_values[0, :], 0):
            raise ValueError("Unknown boundary at S=0 and a(t, 0) not zero for all t is not implemented.")
        if not np.allclose(b_values[0, :], 0):
            raise ValueError("Unknown boundary at S=0 and b(t, 0) not zero for all t is not implemented.")

    # Initialize solution array and apply boundary/final conditions
    V = np.zeros((M + 2, N + 2), dtype=float)
    V[:, -1] = F(S)  # Final condition V(T, S)
    if known_boundaries == 'Both':  # Both boundaries known
        S0_func = kwargs.get('S0_func')
        Smax_func = kwargs.get('Smax_func')
        if S0_func is None or Smax_func is None:
            raise ValueError("For 'Both' known_boundaries, provide S0_func and Smax_func.")
        V = S0_func(t, V)          # Boundary at S_min
        V = Smax_func(t, V)       # Boundary at S_max
    elif known_boundaries == 'S0':  # V(t, S_min) known, V(t, S_max) unknown
        S0_func = kwargs.get('S0_func')
        if S0_func is None:
            raise ValueError("For 'S0' known_boundaries, provide S0_func.")
        V = S0_func(t, V)          # Boundary at S_min
    elif known_boundaries == 'Smax':  # V(t, S_max) known, V(t, S_min) unknown
        Smax_func = kwargs.get('Smax_func')
        if Smax_func is None:
            raise ValueError("For 'Smax' known_boundaries, provide Smax_func.")
        V = Smax_func(t, V)       # Boundary at S_max

    # Preallocate arrays for tridiagonal matrix and RHS to avoid repeated allocations
    M = M
    A_diag = np.empty(M, dtype=float)
    A_lower = np.empty(M - 1, dtype=float)
    A_upper = np.empty(M - 1, dtype=float)

    B_diag = np.empty(M, dtype=float)
    B_lower = np.empty(M - 1, dtype=float)
    B_upper = np.empty(M - 1, dtype=float)

    rhs = np.empty(M, dtype=float)

    # Backward time-stepping loop: j indexes time levels (from N down to 0)
    for j in range(N, -1, -1):

        # Compute time-slice coefficients (vectors over S)
        eta_j = alpha * a_values[:, j]                    # length M+2
        phi_j = 2.0 * alpha * a_values[:, j] + beta * b_values[:, j] - c_values[:, j]
        psi_j = alpha * a_values[:, j] + beta * b_values[:, j]

        # Extract interior node coefficients for i = 1..M (size M)
        eta_i = eta_j[1:M+1]
        phi_i = phi_j[1:M+1]
        psi_i = psi_j[1:M+1]

        # Assemble tridiagonal matrix A coefficients
        # A_diag[i-1] corresponds to node i (i = 1..M)
        A_diag[:] = -gamma - theta * phi_i
        A_lower[:] = theta * eta_i[1:]    # subdiagonal entries for rows 2..M
        A_upper[:] = theta * psi_i[:-1]   # superdiagonal entries for rows 1..M-1

        # Assemble matrix B coefficients
        B_diag[:] = -gamma + one_minus_theta * phi_i
        B_lower[:] = -one_minus_theta * eta_i[1:]
        B_upper[:] = -one_minus_theta * psi_i[:-1]

        # Apply the special rows adjustments of A and B if necessary
        if known_boundaries == 'Both':  # Both boundaries known
            pass
        elif known_boundaries == 'S0':  # V(t, S_min) known, V(t, S_max) unknown
            aux1 = (eta_j[M] - psi_j[M])
            aux2 = (phi_j[M] - 2.0 * psi_j[M])

            A_lower[-1] = theta * aux1
            A_diag[-1] = -gamma - theta * aux2
            B_lower[-1] = -one_minus_theta * aux1
            B_diag[-1] = -gamma + one_minus_theta * aux2
        elif known_boundaries == 'Smax': # V(t, S_max) known, V(t, S_min) unknown
            B_diag[0] = - theta*eta_j[1] * (1 + c[0,j]*dt*one_minus_theta)/(1 - c[0,j]*dt*theta) - one_minus_theta*eta_j[1]
        elif known_boundaries == 'None': # Both boundaries unknown
            aux1 = (eta_j[M] - psi_j[M])
            aux2 = (phi_j[M] - 2.0 * psi_j[M])

            A_lower[-1] = theta * aux1
            A_diag[-1] = -gamma - theta * aux2
            B_lower[-1] = -one_minus_theta * aux1
            B_diag[-1] = -gamma + one_minus_theta * aux2
            B_diag[0] = - theta*eta_j[1] * (1 + c[0,j]*dt*one_minus_theta)/(1 - c[0,j]*dt*theta) - one_minus_theta*eta_j[1]


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
        if known_boundaries == 'Both':  # Both boundaries known
            rhs[0] += - theta * eta_j[1] * V[0, j] - one_minus_theta * eta_j[1] * V[0, j + 1]
            rhs[-1] += theta * psi_j[M] * V[-1, j] - one_minus_theta * psi_j[M] * V[-1, j + 1]
        elif known_boundaries == 'S0':  # V(t, S_min) known, V(t, S_max) unknown
            rhs[0] += - theta * eta_j[1] * V[0, j] - one_minus_theta * eta_j[1] * V[0, j + 1]
        elif known_boundaries == 'Smax': # V(t, S_max) known, V(t, S_min) unknown
            rhs[0] += -theta*eta_j[1] * (dt)/(1 - c[0,j]*dt*theta) * f_values[0,j]
        elif known_boundaries == 'None': # Both boundaries unknown
            rhs[0] += - theta * eta_j[1] * V[0, j] - one_minus_theta * eta_j[1] * V[0, j + 1] - theta*eta_j[1] * (dt)/(1 - c[0,j]*dt*theta) * f_values[0,j]

        # Solve tridiagonal system A * x = rhs (Thomas algorithm)
        V[1:-1, j] = _thomas_solver(A_lower, A_diag, A_upper, rhs)

    # Boundary values adjustments if necessary
    if known_boundaries == 'Both':
        pass
    elif known_boundaries == 'S0':
        V[-1, :-1] = 2.0 * V[-2, :-1] - V[-3, :-1]
    elif known_boundaries == 'Smax':
        V[0, :-1] = (1 + c[0,j]*dt*one_minus_theta)/(1 - c[0,j]*dt*theta) * V[0, 1:] + (dt)/(1 - c[0,j]*dt*theta) * f_values[0, :-1]
    elif known_boundaries == 'None':
        V[-1, :-1] = 2.0 * V[-2, :-1] - V[-3, :-1]
        V[0, :-1] = (1 + c[0,j]*dt*one_minus_theta)/(1 - c[0,j]*dt*theta) * V[0, 1:] + (dt)/(1 - c[0,j]*dt*theta) * f_values[0, :-1]

    return S_mesh, t_mesh, V





