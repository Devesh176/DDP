"""
Transient PIELM — Case 1: 2D heat conduction with internal heat generation
===========================================================================

STEADY PDE (what you already had):
    k*(u_xx + u_yy) + q = 0
    =>  u_xx + u_yy = -q/k

TRANSIENT PDE (what changes):
    rho*cp * dT/dt = k*(T_xx + T_yy) + q

Rearranged and normalised  (u = (T - T_ref) / T_scale):
    u_xx + u_yy  -  (rho*cp/k)*u_t  =  -q/(k*T_scale)
    ──────────────┬──────────────────   ──────┬──────────
    same Laplacian│                     same S_norm as steady
                  │
              NEW time-derivative term

Boundary Conditions — identical to steady (Dirichlet):
    Left  (x=0)   : T = 700 °C  → u = 1
    Right (x=L)   : T = 500 °C  → u = 0
    Bottom(y=0)   : T = 500 °C  → u = 0
    Top   (y=L)   : T = 500 °C  → u = 0

NEW — Initial Condition:
    T(x, y, t_min) = T_initial   (float or callable)

Changes vs steady PIELM_HeatGen
────────────────────────────────
1. W gains a 3rd column  (time weights, own scale st = 2√3/t_span)
2. Input X has 3 columns [x, y, t] instead of [x, y]
3. _H_laplacian → _H_pde : adds  - alpha*phi'(z)*W[:,2]  term
4. K_pde = S_norm  (unchanged — still non-zero due to heat generation)
5. BC points repeated at every time level  (loop over t_e)
6. IC rows added at t = t_min  (plain H value rows)
"""

import numpy as np
import matplotlib.pyplot as plt


def cheby(n, lo, hi):
    """n Chebyshev interior nodes on [lo, hi]."""
    if n < 1:
        return np.array([])
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))


class PIELM_HeatGen_Transient:
    """
    Transient PIELM with internal heat generation.

    Solves:  u_xx + u_yy  -  alpha*u_t  =  S_norm
    where    alpha  = rho*cp / k
             S_norm = -q / (k * T_scale)   [non-zero, same sign as steady]
    """

    def __init__(self, hidden_nodes=1800):
        self.N  = hidden_nodes
        self.W  = self.b = self.c = None
        self._T_ref = self._T_scale = None

    # ── activations ───────────────────────────────────────────────────────────
    def _phi(self, z):      return np.tanh(z)
    def _phi_p(self, z):    return 1.0 - np.tanh(z)**2        # tanh' — NEW
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)                            # tanh'' — same

    def _z(self, X):   return X @ self.W.T + self.b.T
    def _H(self, X):   return self._phi(self._z(X))

    def _H_pde(self, X, alpha):
        """
        PDE rows for:   u_xx + u_yy  -  alpha*u_t  =  S_norm

        Steady _H_laplacian:   phi''(z) * (W0² + W1²)
        Transient adds:      - alpha * phi'(z) * W2     ← time derivative row

        W[:,0]=x-weights, W[:,1]=y-weights, W[:,2]=t-weights (NEW 3rd column)
        """
        z      = self._z(X)
        phi_p  = self._phi_p(z)
        phi_pp = self._phi_pp(z)
        H_lap = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)  # Laplacian rows
        H_t   = phi_p  * self.W[:, 2]                          # time deriv rows
        return H_lap - alpha * H_t

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, nx=40, ny=40, nt=15,
              x_max=0.01, y_max=0.01,
              t_min=0.0,  t_max=100.0,
              T_left=700., T_other=500.,
              k=1000., q=1e9,
              rho=7850., cp=486.,
              T_initial=500.,
              seed=42):
        """
        Parameters
        ----------
        nt        : number of Chebyshev time collocation nodes
        t_min     : start of time domain (set to first ANSYS snapshot if no t=0)
        t_max     : end of time domain
        T_initial : float or callable(x,y) — temperature at t = t_min [°C]
        rho, cp   : density [kg/m³] and specific heat [J/kg-K]
        k, q      : conductivity [W/m-K] and heat generation [W/m³]
        """
        # normalisation — unchanged from steady
        self._T_ref   = T_other
        self._T_scale = T_left - T_other

        def norm(T):
            return (T - self._T_ref) / self._T_scale

        alpha  = rho * cp / k
        S_norm = (-q / k) / self._T_scale   # same as steady, non-zero

        print(f"[Case 1] alpha={alpha:.3f} s/m²,  S_norm={S_norm:.2f}")

        # random weights — NOW 3 columns (x, y, t)
        sx = 2.0*np.sqrt(3.0) / x_max
        sy = 2.0*np.sqrt(3.0) / y_max
        st = 2.0*np.sqrt(3.0) / max(t_max - t_min, 1e-6)  # own time scale

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([
                np.random.uniform(-sx, sx, self.N),
                np.random.uniform(-sy, sy, self.N),
                np.random.uniform(-st, st, self.N),   # NEW 3rd column
            ])
            b_ = np.random.uniform(-1.0, 1.0, (self.N, 1))
            Xs       = np.random.rand(100, 3)
            Xs[:,0] *= x_max;  Xs[:,1] *= y_max
            Xs[:,2]  = Xs[:,2]*(t_max - t_min) + t_min
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        print(f"[Case 1] Condition number: {best_cond:.2e}")

        # PDE rows — interior (x,y,t) Chebyshev grid
        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)

        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.full((len(X_phys), 1), S_norm)   # non-zero: heat generation

        # BC rows — Dirichlet, repeated at every time level
        # Steady: one set of BC rows
        # Transient: same BC rows, but looped over all time levels in t_e
        n_bc = max(nx, ny) * 2
        xe   = np.linspace(0.0, x_max, n_bc)
        ye   = np.linspace(0.0, y_max, n_bc)
        t_e  = np.linspace(t_min, t_max, nt + 2)

        X_bc, K_bc = [], []
        for ti in t_e:
            for yi in ye:
                X_bc.append([0.0,   yi, ti]);  K_bc.append([norm(T_left)])
            for yi in ye:
                X_bc.append([x_max, yi, ti]);  K_bc.append([norm(T_other)])
            for xi in xe[1:-1]:
                X_bc.append([xi, 0.0,   ti]);  K_bc.append([norm(T_other)])
            for xi in xe[1:-1]:
                X_bc.append([xi, y_max, ti]);  K_bc.append([norm(T_other)])

        X_bc = np.array(X_bc);  K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        # IC rows — NEW: spatial domain at t = t_min, plain H value rows
        xc_ic = np.linspace(0.0, x_max, nx)
        yc_ic = np.linspace(0.0, y_max, ny)
        Xg_ic, Yg_ic = np.meshgrid(xc_ic, yc_ic)
        X_ic = np.column_stack((Xg_ic.ravel(), Yg_ic.ravel(),
                                np.full(Xg_ic.size, t_min)))
        H_ic = self._H(X_ic)

        if callable(T_initial):
            T_ic_raw = T_initial(X_ic[:, 0], X_ic[:, 1])
        elif isinstance(T_initial, np.ndarray):
            T_ic_raw = T_initial.ravel()
        else:
            T_ic_raw = np.full(len(X_ic), float(T_initial))
        K_ic = norm(T_ic_raw).reshape(-1, 1)      # must be normalised

        # assemble and solve
        H = np.vstack([H_pde, H_bc, H_ic])
        K = np.vstack([K_pde, K_bc, K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        print(f"[Case 1] PDE:{len(X_phys)}, BC:{len(X_bc)}, IC:{len(X_ic)}, "
              f"N*:{self.N}")
        print(f"[Case 1] System:{H.shape}, rank={rank}, "
              f"smallest SV={sv[-1]:.3e}")
        return self

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, X):
        """X : (n,3) columns [x, y, t].  Returns °C."""
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    L = 0.01; k = 1000.; q = 1e9; rho = 7850.; cp = 486.

    model = PIELM_HeatGen_Transient(hidden_nodes=5000)
    model.train(nx=22, ny=22, nt=15,
                x_max=L, y_max=L,
                t_min=0.0, t_max=60.0,
                T_left=700., T_other=500.,
                k=k, q=q, rho=rho, cp=cp,
                T_initial=500.)

    # left BC at t=30 should be 700
    X_l = np.column_stack((np.zeros(5), np.linspace(0.001,0.009,5), np.full(5,30.)))
    print(f"\nLeft BC  t=30s (expect 700): {model.predict(X_l).round(1)}")

    # right BC at t=30 should be 500
    X_r = np.column_stack((np.full(5,L), np.linspace(0.001,0.009,5), np.full(5,30.)))
    print(f"Right BC t=30s (expect 500): {model.predict(X_r).round(1)}")

    # IC at t=0 should be ~500
    X_ic = np.column_stack((np.full(5,0.005), np.linspace(0.001,0.009,5), np.zeros(5)))
    print(f"IC       t=0s  (expect 500): {model.predict(X_ic).round(1)}")

    # max T should exceed 700 due to heat generation
    xp = np.linspace(0,L,50); yp = np.linspace(0,L,50)
    Xg,Yg = np.meshgrid(xp,yp)
    X_ev = np.column_stack((Xg.ravel(), Yg.ravel(), np.full(2500,50.)))
    T50 = model.predict(X_ev)
    print(f"T range at t=50s: {T50.min():.1f} – {T50.max():.1f} °C "
          f"(max should exceed 700)")

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(f"Case 1 transient  q={q:.0e} W/m³  k={k} W/m-K")
    for ax, t_s in zip(axes, [1., 20., 55.]):
        X_ev = np.column_stack((Xg.ravel(), Yg.ravel(), np.full(2500,t_s)))
        T = model.predict(X_ev).reshape(50,50)
        cf = ax.contourf(Xg*1e3, Yg*1e3, T, levels=40, cmap='hot')
        fig.colorbar(cf, ax=ax, label='T (°C)')
        ax.set_title(f't={t_s}s  [{T.min():.0f}–{T.max():.0f}°C]')
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('case1_transient_result.png', dpi=150, bbox_inches='tight')
    print("Saved case1_transient_result.png")