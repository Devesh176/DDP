"""
PIELM — 2D transient heat conduction (baseline, Dirichlet BCs)
==============================================================
Governing PDE:
    rho*cp * dT/dt  =  k * (d2T/dx2 + d2T/dy2)

Rearranged for the linear system (normalised u = (T-T_ref)/T_scale):
    u_xx + u_yy  -  (rho*cp/k) * u_t  =  0

Three sets of constraint rows assembled into  H·c = K:
    1. PDE rows      — interior (x,y,t) collocation points
    2. BC  rows      — boundary edges at every time level
    3. IC  rows      — entire spatial domain at t = 0
"""

import numpy as np


def cheby(n, lo, hi):
    """n Chebyshev interior nodes on [lo, hi]."""
    if n < 1:
        return np.array([])
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))


class PIELM2DDiffusion_transient:
    """
    PIELM for 2D transient heat conduction with Dirichlet BCs.
    Direct solve: H·c = K via lstsq pseudoinverse — no iteration.
    """

    def __init__(self, hidden_nodes=2000):
        self.hidden_nodes = hidden_nodes
        self.W = self.b = self.c = None
        self._T_ref = self._T_scale = None

    # ── activations ──────────────────────────────────────────────────────────
    def _phi(self, z):      return np.tanh(z)
    def _phi_p(self, z):    return 1.0 - np.tanh(z)**2        # tanh'
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)                            # tanh''

    def _z(self, X):   return X @ self.W.T + self.b.T
    def _H(self, X):   return self._phi(self._z(X))

    def _H_pde(self, X, alpha):
        """
        PDE rows for:  u_xx + u_yy  -  alpha * u_t  =  0

        d2f/dx2  =  phi''(z) * W[:,0]^2
        d2f/dy2  =  phi''(z) * W[:,1]^2
        df/dt    =  phi'(z)  * W[:,2]      <-- phi_PRIME, not phi

        Row = phi''*(W0^2 + W1^2)  -  alpha * phi'*W2
        """
        z      = self._z(X)
        phi_p  = self._phi_p(z)                  # (n, N*)
        phi_pp = self._phi_pp(z)                 # (n, N*)
        H_lap  = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)
        H_t    = phi_p  * self.W[:, 2]           # BUG 1 FIX: phi_p not phi
        return H_lap - alpha * H_t

    """
Fixes for test_2d_diffusion_transient_case1.py
===============================================

Three changes needed in the train() call:

1. T_initial = 500.0  (not 22.0)
   The plate starts at T_other = 500°C before the left edge heats up.
   22°C is room temperature which is wrong for this problem.

2. t_min = t_data_min (start time grid from your first data point)
   Your ANSYS data starts at t=3.0s, not t=0.
   Either:
     a) include t=0 ANSYS file in your data folder, OR
     b) train on t_min=3.0 to t_max=9.82 with the IC at t=3.0s

3. hidden_nodes = 5000  (not 1800)
   Rule: N* ~ Nf + Nbc + Nic ~ 21392 rows → need ~5000 neurons minimum.

OPTION A — You have a t=0 ANSYS file (recommended):
    Add it to your data folder named like: transient_case1_temp_0.0.xls
    Then train with:
        model.train(
            nx=25, ny=25, nt=15,
            x_max=x_max, y_max=y_max,
            t_max=t_max,            # now starts from 0
            T_left=700., T_other=500.,
            T_initial=500.,         # FIX 1: correct IC
            hidden_nodes=5000,      # FIX 2: enough neurons
            rho=..., cp=..., k=...
        )

OPTION B — You only have t=3.0s to 9.82s (use data-driven IC):
    Extract the temperature field at t=3.0s as your IC,
    and train on the interval [t_min, t_max]:

        t_min = df['t'].min()   # 3.0s
        t_max = df['t'].max()   # 9.82s

        # Build IC from ANSYS data at t=t_min
        df_ic = df[df['t'] == t_min][['x', 'y', 'temp']].copy()

        # Interpolate IC onto a regular grid
        from scipy.interpolate import griddata
        xg = np.linspace(0, x_max, 25)
        yg = np.linspace(0, y_max, 25)
        Xg, Yg = np.meshgrid(xg, yg)
        T_ic_grid = griddata(
            df_ic[['x','y']].values,
            df_ic['temp'].values,
            (Xg, Yg),
            method='linear',
            fill_value=500.
        )
        # Pass as callable to train()
        from scipy.interpolate import RegularGridInterpolator
        ic_interp = RegularGridInterpolator(
            (xg, yg), T_ic_grid.T,
            method='linear', bounds_error=False, fill_value=500.
        )
        T_initial_fn = lambda x, y: ic_interp(np.column_stack([x, y]))

        model = PIELM2DDiffusion_transient(hidden_nodes=5000)
        model.train(
            nx=25, ny=25, nt=15,
            x_max=x_max, y_max=y_max,
            t_min=t_min,            # FIX 3: start time axis at 3.0s
            t_max=t_max,
            T_left=700., T_other=500.,
            T_initial=T_initial_fn, # FIX 1: data-driven IC
            rho=..., cp=..., k=...
        )

NOTE: The train() function needs a small update to accept t_min:

    def train(self, ..., t_min=0.0, t_max=100.0, ...):
        ...
        tc = cheby(nt, t_min, t_max)    # time grid starts at t_min
        t_e = np.linspace(t_min, t_max, nt+2)   # BC time levels
        X_ic has t = t_min (not t = 0)
"""

# # ── Minimal patch to add t_min support to the existing class ─────────────

# import numpy as np
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))
# from _2d_diffusion_transient import PIELM2DDiffusion_transient, cheby


# class PIELM2DDiffusion_transient_v2(PIELM2DDiffusion_transient):
#     """
#     Extends the base class with:
#       - t_min parameter (so time axis can start at first ANSYS snapshot)
#       - T_initial as scalar, callable, or ndarray
#     """

    def train(self, nx=20, ny=20, nt=10,
              x_max=0.01, y_max=0.01,
              t_min=0.0, t_max=100.0,
              rho=7850.0, cp=486.0, k=45.0,
              T_left=700.0, T_other=500.0,
              T_initial=500.0,
              seed=42):

        self._T_ref   = T_other
        self._T_scale = T_left - T_other
        alpha = rho * cp / k

        def normalise(T):
            return (T - self._T_ref) / self._T_scale

        # weights
        sx = 2.0*np.sqrt(3.0) / x_max
        sy = 2.0*np.sqrt(3.0) / y_max
        st = 2.0*np.sqrt(3.0) / max(t_max - t_min, 1e-6)

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([
                np.random.uniform(-sx, sx, self.hidden_nodes),
                np.random.uniform(-sy, sy, self.hidden_nodes),
                np.random.uniform(-st, st, self.hidden_nodes),
            ])
            b_ = np.random.uniform(-1.0, 1.0, (self.hidden_nodes, 1))
            Xs = np.random.rand(100, 3)
            Xs[:, 0] *= x_max; Xs[:, 1] *= y_max
            Xs[:, 2]  = Xs[:, 2] * (t_max - t_min) + t_min
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        print(f"Best condition number: {best_cond:.2e}")

        # PDE collocation — time axis from t_min to t_max
        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)          # KEY FIX: starts at t_min
        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.zeros((len(X_phys), 1))

        # BC rows — time from t_min to t_max
        n_bc = max(nx, ny) * 2
        x_e  = np.linspace(0.0, x_max, n_bc)
        y_e  = np.linspace(0.0, y_max, n_bc)
        t_e  = np.linspace(t_min, t_max, nt + 2)   # KEY FIX

        X_bc, K_bc = [], []
        for ti in t_e:
            for yi in y_e:
                X_bc.append([0.0,   yi, ti]);  K_bc.append([normalise(T_left)])
            for yi in y_e:
                X_bc.append([x_max, yi, ti]);  K_bc.append([normalise(T_other)])
            for xi in x_e[1:-1]:
                X_bc.append([xi, 0.0,   ti]);  K_bc.append([normalise(T_other)])
            for xi in x_e[1:-1]:
                X_bc.append([xi, y_max, ti]);  K_bc.append([normalise(T_other)])

        X_bc = np.array(X_bc); K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        # IC rows — evaluated at t = t_min (not necessarily t = 0)
        xc_ic = np.linspace(0.0, x_max, nx)
        yc_ic = np.linspace(0.0, y_max, ny)
        Xg_ic, Yg_ic = np.meshgrid(xc_ic, yc_ic)
        # X_ic = np.column_stack((Xg_ic.ravel(), Yg_ic.ravel(),
        #                         np.full(Xg_ic.size, t_min)))  # KEY FIX: t_min
        # fixed — exclude left boundary column from IC
        mask = Xg_ic.ravel() > x_max / nx
        X_ic = np.column_stack((Xg_ic.ravel()[mask], Yg_ic.ravel()[mask],
                                np.full(mask.sum(), t_min)))

        H_ic = self._H(X_ic)

        if callable(T_initial):
            T_ic_raw = T_initial(X_ic[:, 0], X_ic[:, 1])
        elif isinstance(T_initial, np.ndarray):
            T_ic_raw = T_initial.ravel()
        else:
            T_ic_raw = np.full(len(X_ic), float(T_initial))
        K_ic = normalise(T_ic_raw).reshape(-1, 1)


        K_ic = K_ic.flatten()  # Convert (1560, 1) to (1560,)
        mask = mask[:len(K_ic)]  # Trim mask to match
        K_ic = K_ic[mask]

        H = np.vstack([H_pde, H_bc, H_ic])
        K = np.vstack([K_pde, K_bc, K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        print(f"PDE: {len(X_phys)}, BC: {len(X_bc)}, IC: {len(X_ic)}, N*: {self.hidden_nodes}")
        print(f"System: {H.shape}, rank={rank}, smallest SV={sv[-1]:.3e}")
        return self

    # ── prediction ───────────────────────────────────────────────────────────
    def predict(self, X, return_celsius=True):
        """
        X : ndarray (n, 3) — columns [x, y, t]
        Returns temperature in °C (default) or normalised u.
        """
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref if return_celsius else u