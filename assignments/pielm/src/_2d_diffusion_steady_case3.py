"""
Case 2 — 2D Steady Heat Conduction with Mixed Boundary Conditions
=================================================================
Governing PDE (no heat generation):
    u_xx + u_yy = 0

Boundary Conditions:
    Left  edge (x=0)   : heat flux   -k·du/dx = q_flux  => du/dx = -q_flux/k
    Top   edge (y=L)   : convection  -k·du/dy = h·(T - T_inf)
                                     => -k·du/dy - h·T = -h·T_inf
    Right edge (x=L)   : insulated   du/dx = 0
    Bottom edge (y=0)  : insulated   du/dy = 0

Material:
    k      = 1000  W/m-K
    h      = 25    W/m²-K  (convection coefficient, top edge)
    q_flux = 500   W/m²    (heat flux into plate, left edge)
    T_inf  = 22    °C      (ambient temperature, convection)

─────────────────────────────────────────────────────────────────
NORMALISATION NOTE
─────────────────────────────────────────────────────────────────
No Dirichlet BC fixes an absolute temperature here, so we work
directly in °C (no normalisation).  The system is well-posed
because the convection BC anchors the absolute temperature level
via T_inf.

─────────────────────────────────────────────────────────────────
BOUNDARY CONDITION DERIVATION (how each BC becomes an H row)
─────────────────────────────────────────────────────────────────

1. LEFT EDGE — Neumann (heat flux inward, x=0):
   Physics:  -k · dT/dx|_{x=0} = q_flux
   =>         dT/dx|_{x=0}     = -q_flux / k
   PIELM row: H_x · c = -q_flux / k
   where H_x[i,j] = phi'(z_j) * W[j,0]    (derivative w.r.t. x)

2. TOP EDGE — Robin/convection (y=L):
   Physics:  -k · dT/dy|_{y=L} = h·(T - T_inf)
   =>        -k · dT/dy - h·T  = -h·T_inf
   PIELM row: (-k·H_y - h·H) · c = -h·T_inf
   where H_y[i,j] = phi'(z_j) * W[j,1]    (derivative w.r.t. y)
   RHS = -h·T_inf  (scalar, same at every top-edge point)

3. RIGHT EDGE — Neumann/insulated (x=L):
   Physics:  dT/dx|_{x=L} = 0
   PIELM row: H_x · c = 0

4. BOTTOM EDGE — Neumann/insulated (y=0):
   Physics:  dT/dy|_{y=0} = 0
   PIELM row: H_y · c = 0
"""

import numpy as np
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def cheby(n, lo, hi):
    """Chebyshev interior nodes."""
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))


class PIELM_MixedBC:
    """
    PIELM for 2D Laplace with Neumann, Robin, and insulated BCs.
    Works in physical °C — no normalisation needed (convection anchors T).
    """

    def __init__(self, hidden_nodes=1800):
        self.N = hidden_nodes
        self.W = self.b = self.c = None

    # ── activation and derivatives ───────────────────────────────────────────
    def _phi(self, z):      return np.tanh(z)
    def _phi_p(self, z):    return 1 - np.tanh(z)**2          # tanh'(z)
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2*t*(1 - t**2)                                 # tanh''(z)

    def _z(self, X):   return X @ self.W.T + self.b.T

    def _H(self, X):
        """Value rows: f = H·c"""
        return self._phi(self._z(X))

    def _H_dx(self, X):
        """
        Derivative rows for df/dx = H_dx · c
        df/dx = sum_j c_j * phi'(z_j) * W[j,0]
        => row j = phi'(z_j) * W[j,0]
        """
        return self._phi_p(self._z(X)) * self.W[:, 0]

    def _H_dy(self, X):
        """
        Derivative rows for df/dy = H_dy · c
        df/dy = sum_j c_j * phi'(z_j) * W[j,1]
        => row j = phi'(z_j) * W[j,1]
        """
        return self._phi_p(self._z(X)) * self.W[:, 1]

    def _H_laplacian(self, X):
        """PDE rows: d²f/dx² + d²f/dy²"""
        z = self._z(X)
        return self._phi_pp(z) * (self.W[:, 0]**2 + self.W[:, 1]**2)

    # ── training ─────────────────────────────────────────────────────────────
    def train(self, nx=40, ny=40,
              x_max=0.01, y_max=0.01,
              k=1000., h=25., q_flux=500., T_inf=22.,
              seed=42):

        self.k = k;  self.h = h
        self.q_flux = q_flux;  self.T_inf = T_inf

        # --- random weights --------------------------------------------------
        # Working in °C means temperatures are O(100), so we need a slightly
        # different scaling reference.  We still want W·x ~ O(1).
        sx = 2*np.sqrt(3) / x_max
        sy = 2*np.sqrt(3) / y_max

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([
                np.random.uniform(-sx, sx, self.N),
                np.random.uniform(-sy, sy, self.N),
            ])
            b_ = np.random.uniform(-1, 1, (self.N, 1))
            Xs = np.random.rand(100, 2) * [x_max, y_max]
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        print(f"[Case 2] Hidden layer condition number: {best_cond:.2e}")

        # --- PDE rows (interior) ---------------------------------------------
        xc = cheby(nx-2, 0., x_max)
        yc = cheby(ny-2, 0., y_max)
        Xg, Yg = np.meshgrid(xc, yc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel()))

        H_pde = self._H_laplacian(X_phys)
        K_pde = np.zeros((len(X_phys), 1))       # Laplace: RHS = 0

        # --- BC rows ---------------------------------------------------------
        n_bc = max(nx, ny) * 2
        xe = np.linspace(0., x_max, n_bc)
        ye = np.linspace(0., y_max, n_bc)

        rows_H, rows_K = [], []

        # 1. LEFT EDGE (x=0) — heat flux inward
        #    -k · du/dx = q_flux  =>  du/dx = -q_flux/k
        #    H_dx · c = -q_flux/k
        X_left = np.column_stack((np.zeros(n_bc), ye))
        rhs_left = -q_flux / k                           # scalar
        rows_H.append(self._H_dx(X_left))
        rows_K.append(np.full((n_bc, 1), rhs_left))

        # 2. TOP EDGE (y=L) — convection
        #    -k·du/dy - h·T = -h·T_inf
        #    (-k·H_dy - h·H) · c = -h·T_inf
        X_top = np.column_stack((xe, np.full(n_bc, y_max)))
        H_top_conv = -k * self._H_dy(X_top) - h * self._H(X_top)
        rhs_top = -h * T_inf                             # scalar
        rows_H.append(H_top_conv)
        rows_K.append(np.full((n_bc, 1), rhs_top))

        # 3. RIGHT EDGE (x=L) — insulated  du/dx = 0
        X_right = np.column_stack((np.full(n_bc, x_max), ye))
        rows_H.append(self._H_dx(X_right))
        rows_K.append(np.zeros((n_bc, 1)))

        # 4. BOTTOM EDGE (y=0) — insulated  du/dy = 0
        X_bot = np.column_stack((xe, np.zeros(n_bc)))
        rows_H.append(self._H_dy(X_bot))
        rows_K.append(np.zeros((n_bc, 1)))

        # --- assemble and solve ----------------------------------------------
        H = np.vstack([H_pde] + rows_H)
        K = np.vstack([K_pde] + rows_K)

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        print(f"[Case 2] System: {H.shape}, rank={rank}, "
              f"smallest SV={sv[-1]:.3e}")
        return self

    # ── prediction ───────────────────────────────────────────────────────────
    def predict(self, X):
        """Return temperature in °C at points X (n,2)."""
        return (self._H(X) @ self.c).ravel()

    def predict_flux_x(self, X):
        """Return heat flux in x-direction: qx = -k·dT/dx  [W/m²]"""
        return -self.k * (self._H_dx(X) @ self.c).ravel()

    def predict_flux_y(self, X):
        """Return heat flux in y-direction: qy = -k·dT/dy  [W/m²]"""
        return -self.k * (self._H_dy(X) @ self.c).ravel()


