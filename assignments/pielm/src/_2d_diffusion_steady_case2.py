"""
Case 1 — 2D Steady Heat Conduction with Internal Heat Generation
================================================================
Governing PDE:
    k*(u_xx + u_yy) + q = 0
    =>  u_xx + u_yy = -q/k

Boundary Conditions (Dirichlet):
    Left  edge (x=0)   : T = 700 °C
    Right edge (x=L)   : T = 500 °C
    Bottom edge (y=0)  : T = 500 °C
    Top   edge (y=L)   : T = 500 °C

Material:
    k = 1000 W/m-K
    q = 1e9  W/m³
    Source term: -q/k = -1e9/1000 = -1e6  (in normalised u space, see below)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# ── helpers ──────────────────────────────────────────────────────────────────

def cheby(n, lo, hi):
    """Chebyshev interior nodes — clusters points near boundaries."""
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))


class PIELM_HeatGen:
    """
    PIELM for  u_xx + u_yy = S  (S = -q/k, a constant source term).

    Temperature is normalised as:
        u = (T - T_ref) / T_scale
    so all numbers in the linear system stay O(1).

    After normalising, the source term becomes:
        S_norm = S / T_scale  =  (-q/k) / T_scale
    because  d²(u·T_scale + T_ref)/dx² = T_scale · d²u/dx²
    => T_scale·(u_xx + u_yy) = S
    => u_xx + u_yy = S / T_scale
    """

    def __init__(self, hidden_nodes=1800):
        self.N = hidden_nodes
        self.W = self.b = self.c = None
        self._T_ref = self._T_scale = None

    # ── activation and derivatives ───────────────────────────────────────────
    def _phi(self, z):          return np.tanh(z)
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2*t*(1 - t**2)  # tanh''(z)

    def _z(self, X):   return X @ self.W.T + self.b.T
    def _H(self, X):   return self._phi(self._z(X))

    def _H_laplacian(self, X):
        """
        Matrix rows for  d²f/dx² + d²f/dy² = S_norm.

        d²f/dx² = phi''(z) * W[:,0]²
        d²f/dy² = phi''(z) * W[:,1]²
        => row = phi''(z) * (W[:,0]² + W[:,1]²)
        """
        z = self._z(X)
        return self._phi_pp(z) * (self.W[:, 0]**2 + self.W[:, 1]**2)

    # ── training ─────────────────────────────────────────────────────────────
    def train(self, nx=40, ny=40,
              x_max=0.01, y_max=0.01,
              T_left=700., T_other=500.,
              k=1000., q=1e9,
              seed=42):

        # --- normalisation ---------------------------------------------------
        self._T_ref   = T_other
        self._T_scale = T_left - T_other          # 200 °C

        # Source term in normalised coordinates
        S_norm = (-q / k) / self._T_scale         # = -1e6 / 200 = -5000

        # --- random weights: scale so W·x ~ O(1) over domain ----------------
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
        print(f"[Case 1] Hidden layer condition number: {best_cond:.2e}")

        # --- PDE collocation rows (interior Chebyshev grid) ------------------
        xc = cheby(nx-2, 0., x_max)
        yc = cheby(ny-2, 0., y_max)
        Xg, Yg = np.meshgrid(xc, yc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel()))

        H_pde = self._H_laplacian(X_phys)             # (n_phys, N)
        K_pde = np.full((len(X_phys), 1), S_norm)     # RHS = -q/k normalised
        #
        # KEY DIFFERENCE FROM PURE DIFFUSION:
        # K_pde is NOT zero — it equals S_norm at every collocation point.
        # This encodes the internal heat generation into the linear system.

        # --- BC rows (Dirichlet on all four edges) ---------------------------
        n_bc = max(nx, ny) * 2
        xe = np.linspace(0., x_max, n_bc)
        ye = np.linspace(0., y_max, n_bc)

        X_bc, K_bc = [], []
        # normalised BCs:  u_left=1, u_others=0
        for yi in ye:   X_bc.append([0.,    yi]);  K_bc.append([1.])  # left
        for yi in ye:   X_bc.append([x_max, yi]);  K_bc.append([0.])  # right
        for xi in xe[1:-1]: X_bc.append([xi, 0.]);     K_bc.append([0.])  # bot
        for xi in xe[1:-1]: X_bc.append([xi, y_max]);  K_bc.append([0.])  # top

        X_bc = np.array(X_bc);  K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        # --- assemble and solve H·c = K --------------------------------------
        H = np.vstack([H_pde, H_bc])
        K = np.vstack([K_pde, K_bc])
        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)

        print(f"[Case 1] System: {H.shape}, rank={rank}, "
              f"smallest SV={sv[-1]:.3e}")
        return self

    # ── prediction ───────────────────────────────────────────────────────────
    def predict(self, X):
        """Return temperature in °C at points X (n,2)."""
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref


