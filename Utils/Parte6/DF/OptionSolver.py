import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

from Utils.Parte6.DF.Utils import plot_func, plot_surface, solve_PDE


class OptionSolver(ABC):
    '''
    Abstract base class for finite difference option solvers.
    '''

    name = "Option solver"

    @abstractmethod
    def __init__(self):
        pass

    def solve(self, known_boundaries: str = None):
        '''
        Solve the PDE using the finite difference method.
        This method sets up the grid and calls the generic solver.
        '''
        print(f"[{self.name}]\t\tSolving option...")

        if known_boundaries is None:
            # Determine known boundaries based on presence of apply_S0 and apply_Smax methods
            has_S0 = callable(getattr(self, 'apply_S0', None))
            has_Smax = callable(getattr(self, 'apply_Smax', None))
            if has_S0 and has_Smax:
                self.known_boundaries = 'Both'
            elif has_S0:
                self.known_boundaries = 'S0'
            elif has_Smax:
                self.known_boundaries = 'Smax'
            else:
                self.known_boundaries = 'None'
        else:
            self.known_boundaries = known_boundaries

        
        self.S, self.t, self.V = solve_PDE(
            a=self.a,
            b=self.b,
            c=self.c,
            f=self.f,
            payoff=self.payoff,
            S_min=self.S_min,
            S_max=self.S_max,
            T=self.T,
            known_boundaries=self.known_boundaries,
            N=self.N,
            M=self.M,
            theta=self.theta,
            S0_func=self.apply_S0 if 'S0' in self.known_boundaries else None,
            Smax_func=self.apply_Smax if 'Smax' in self.known_boundaries else None,
        )


    def get_solution(self):
        """
        Return S, t, V without plotting.
        """
        if not hasattr(self, "V"):
            self.solve()
        return self.S, self.t, self.V

    def plot(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the surface V(S, t). If fig/ax are provided, draw on them.
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"Plotting surface: {self.name}...")
        if not hasattr(self, "V"):
            self.solve()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.V, xlabel='S', ylabel='t', zlabel='Value', fig=fig, ax=ax, title=self.title, **surface_kwargs)


    def get_value_at_S(self, S0: float):
        """
        Returns (t, V(S0, t)) as 1D arrays.
        Parameters:
            S0: Asset price at which to evaluate V.
        Returns:
            t, V(S0, t): 1D arrays of time and option value at S0.
        """
        print(f"[{self.name}]\t\tGetting value at S={S0}...")
        if not hasattr(self, "V"):
            self.solve()
        DS = self.S_max / (self.M + 1)
        idx = int(round(S0 / DS))
        return self.t[0, :], self.V[idx, :]

    def plot_value_at_S(self, S0: float, fig=None, ax=None, **plot_kwargs):
        """
        Plot V(S0, t) as a function of t.
        Parameters:
            S0: Asset price at which to evaluate V.
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            plot_kwargs: Extra kwargs forwarded to ax.plot (e.g., color, linestyle).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting value at S={S0}...")
        t, V_S0 = self.get_value_at_S(S0)
        if not hasattr(self, "title"):
            self.title = None
        return plot_func(t, V_S0, xlabel='t', ylabel=f'V(S={S0}, t)', fig=fig, ax=ax, title=self.title, **plot_kwargs)

    def get_value_at_t(self, t0: float):
        """
        Returns (S, V(S, t0)) as 1D arrays.
        Parameters:
            t0: Time at which to evaluate V.
        Returns:
            S, V(S, t0): 1D arrays of asset price and option value at t0.
        """
        print(f"[{self.name}]\t\tGetting value at t={t0}...")
        if not hasattr(self, "V"):
            self.solve()
        Dt = self.T / (self.N + 1)
        j = int(round(t0 / Dt))
        return self.S[:, 0], self.V[:, j]
    
    def plot_value_at_t(self, t0: float, fig=None, ax=None, **plot_kwargs):
        """
        Plot V(S, t0) as a function of S.
        Parameters:
            t0: Time at which to evaluate V.
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            plot_kwargs: Extra kwargs forwarded to ax.plot (e.g., color, linestyle).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting value at t={t0}...")
        S, V_t0 = self.get_value_at_t(t0)
        if not hasattr(self, "title"):
            self.title = None
        return plot_func(S, V_t0, xlabel='S', ylabel=f'V(S, t={t0})', fig=fig, ax=ax, title=self.title, **plot_kwargs)


    def get_delta(self):
        """
        Compute the option delta using the theta-weighted finite difference scheme:
            delta ≈ θ * (V_{j}^{i+1} - V_{j}^{i}) / ΔS + (1 - θ) * (V_{j+1}^{i+1} - V_{j+1}^{i}) / ΔS
        Returns:
            delta: Array of delta values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Delta...")
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
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Delta surface...")
        if not hasattr(self, "delta"):
            self.get_delta()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.delta, xlabel='S', ylabel='t', zlabel='Delta', fig=fig, ax=ax, title='Delta ' + self.title, **surface_kwargs)


    def get_gamma(self):
        """
        Compute the option gamma using the theta-weighted finite difference scheme:
            gamma ≈ θ * (V_{j}^{i+1} - 2V_{j}^{i} + V_{j}^{i-1}) / ΔS^2
                  + (1 - θ) * (V_{j+1}^{i+1} - 2V_{j+1}^{i} + V_{j+1}^{i-1}) / ΔS^2
        Returns:
            gamma: Array of gamma values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Gamma...")
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
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Gamma surface...")
        if not hasattr(self, "gamma"):
            self.get_gamma()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.gamma, xlabel='S', ylabel='t', zlabel='Gamma', fig=fig, ax=ax, title='Gamma ' + self.title, **surface_kwargs)


    def get_theta(self):
        """
        Compute the option theta using the finite difference scheme:
            theta ≈ (V_{j+1}^{i} - V_{j}^{i}) / Δt
        Returns:
            theta: Array of theta values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Theta...")
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
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Theta surface...")
        if not hasattr(self, "theta_arr"):
            self.get_theta()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.theta_arr, xlabel='S', ylabel='t', zlabel='Theta', fig=fig, ax=ax, title='Theta ' + self.title, **surface_kwargs)


    def get_speed(self):
        """
        Compute the option speed (third derivative with respect to S) using the theta-weighted finite difference scheme:
            speed ≈ θ * (-V_{j}^{i-2} + 2V_{j}^{i-1} - 2V_{j}^{i+1} + V_{j}^{i+2}) / (8 ΔS^3)
                    + (1 - θ) * (-V_{j+1}^{i-2} + 2V_{j+1}^{i-1} - 2V_{j+1}^{i+1} + V_{j+1}^{i+2}) / (8 ΔS^3)
        Returns:
            speed: Array of speed values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Speed...")
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
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Speed surface...")
        if not hasattr(self, "speed"):
            self.get_speed()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.speed, xlabel='S', ylabel='t', zlabel='Speed', fig=fig, ax=ax, title='Speed ' + self.title, **surface_kwargs)


    def get_vega(self, dsigma=1e-6):
        """
        Approximate the option vega (∂V/∂σ) using central finite differences.
        Returns:
            vega: Array of vega values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Vega...")
        if self.EDE != 'lognormal':
            raise NotImplementedError("Vega computation only implemented for lognormal model.")

        if not hasattr(self, "V"):
            self.solve()

        sigma_orig = getattr(self, "sigma", 0.2)

        # Save original sigma and V
        V_orig = self.V.copy()

        # Perturb sigma and re-solve
        self.sigma = sigma_orig + dsigma
        self.solve()
        V_p = self.V.copy()

        self.sigma = sigma_orig - dsigma
        self.solve()
        V_m = self.V.copy()

        # Restore original sigma and solution
        self.sigma = sigma_orig
        self.V = V_orig

        vega = (V_p - V_m) / (2 * dsigma)
        self.vega = vega
        return vega

    def plot_vega(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the vega surface.
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Vega surface...")
        if not hasattr(self, "vega"):
            self.get_vega()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.vega, xlabel='S', ylabel='t', zlabel='Vega', fig=fig, ax=ax, title='Vega ' + self.title, **surface_kwargs)


    def get_rho_r(self, dr=1e-6):
        """
        Approximate the option rho (∂V/∂r) using central finite differences.
        Returns:
            rho_r: Array of rho values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Rho (r)...")
        if not hasattr(self, "V"):
            self.solve()

        r_orig = self.r
        V_orig = self.V.copy()

        # r + dr
        self.r = r_orig + dr
        self.solve()
        V_p = self.V.copy()

        # r - dr
        self.r = r_orig - dr
        self.solve()
        V_m = self.V.copy()

        # Restore original state
        self.r = r_orig
        self.V = V_orig

        rho_r = (V_p - V_m) / (2 * dr)
        self.rho_r = rho_r
        return rho_r

    def plot_rho_r(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the rho (with respect to r) surface.
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Rho (r) surface...")
        if not hasattr(self, "rho_r"):
            self.get_rho_r()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.rho_r, xlabel='S', ylabel='t', zlabel='Rho (r)', fig=fig, ax=ax, title='Rho (r) ' + self.title, **surface_kwargs)


    def get_rho_D(self, dD=1e-6):
        """
        Approximate the option rho_D (∂V/∂D) using central finite differences.
        Returns:
            rho_D: Array of rho_D values with shape (M + 2, N + 2)
        """
        print(f"[{self.name}]\t\tComputing Rho (D)...")
        if not hasattr(self, "V"):
            self.solve()

        D_orig = self.D
        V_orig = self.V.copy()

        # D + dD
        self.D = D_orig + dD
        self.solve()
        V_p = self.V.copy()

        # D - dD
        self.D = D_orig - dD
        self.solve()
        V_m = self.V.copy()

        # Restore original state
        self.D = D_orig
        self.V = V_orig

        rho_D = (V_p - V_m) / (2 * dD)
        self.rho_D = rho_D
        return rho_D

    def plot_rho_D(self, fig=None, ax=None, **surface_kwargs):
        """
        Plot the rho (with respect to D) surface.
        Parameters:
            fig, ax: Optional matplotlib Figure and Axes to draw on.
            surface_kwargs: Extra kwargs forwarded to ax.plot_surface (e.g., alpha, cmap).
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        print(f"[{self.name}]\t\tPlotting Rho (D) surface...")
        if not hasattr(self, "rho_D"):
            self.get_rho_D()
        if not hasattr(self, "title"):
            self.title = None
        return plot_surface(self.S, self.t, self.rho_D, xlabel='S', ylabel='t', zlabel='Rho (D)', fig=fig, ax=ax, title='Rho (D) ' + self.title, **surface_kwargs)


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


    # Auxiliary methods
    def apply_payoff(self, S, V):
        '''
        Apply the payoff function to the final time slice.
        Parameters:
            S (array): Stock price grid.
            V (2D array): Option value grid to be modified in place.
        Returns:
            V (2D array): Modified option value grid with payoff applied at maturity.
        '''
        if not hasattr(self, "payoff"):
            raise NotImplementedError("The payoff function must be defined in subclasses.")
        V[:, -1] = np.asarray(self.payoff(S), dtype=float)

        return V
    
    def find_closest_S_index(self, S0):
        '''
        Find the index of the grid point closest to S0.
        Parameters:
            S0 (float or np.ndarray): Stock price(s) to find.
        Returns:
            index (int or np.ndarray): Index/indices of the closest stock price(s) in the grid.
        '''
        DS = self.S_max / (self.M + 1)
        if np.isscalar(S0):
            idx = int(round(S0 / DS))
            if idx < 0 or idx > self.M + 1:
                raise ValueError(f"S0={S0} is out of bounds for the stock price grid [0, {self.S_inf}].")
            return idx
        else:
            idx_arr = np.round(np.asarray(S0) / DS).astype(int)
            if np.any(idx_arr < 0) or np.any(idx_arr > self.M + 1):
                raise ValueError(f"Some S0 values are out of bounds for the stock price grid [0, {self.S_inf}].")
            return idx_arr
        
    def find_closest_t_index(self, t0):
        '''
        Find the index of the grid point closest to t0.
        Parameters:
            t0 (float or np.ndarray): Time(s) to find.
        Returns:
            index (int or np.ndarray): Index/indices of the closest time(s) in the grid.
        '''
        Dt = self.T / (self.N + 1)
        if np.isscalar(t0):
            j = int(round(t0 / Dt))
            if j < 0 or j > self.N + 1:
                raise ValueError(f"t0={t0} is out of bounds for the time grid [0, {self.T}].")
            return j
        else:
            j_arr = np.round(np.asarray(t0) / Dt).astype(int)
            if np.any(j_arr < 0) or np.any(j_arr > self.N + 1):
                raise ValueError(f"Some t0 values are out of bounds for the time grid [0, {self.T}].")
            return j_arr
