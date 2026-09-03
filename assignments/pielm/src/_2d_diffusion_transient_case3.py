"""
Transient PIELM — Case 2: 2D heat conduction with mixed BCs
============================================================

STEADY PDE (what you already had):
    u_xx + u_yy = 0

TRANSIENT PDE (what changes):
    rho*cp * dT/dt = k*(T_xx + T_yy)

Rearranged (working directly in °C — no normalisation, same reason as steady):
    T_xx + T_yy  -  alpha*T_t  =  0      alpha = rho*cp/k

Boundary Conditions — identical to steady Case 2 (mixed):
    Left  (x=0)  : heat flux    -k*dT/dx = q_flux   → dT/dx = -q_flux/k
    Top   (y=L)  : convection   -k*dT/dy = h*(T-T_inf)
                                → (-k*H_dy - h*H)*c  = -h*T_inf
    Right (x=L)  : insulated    dT/dx = 0
    Bottom(y=0)  : insulated    dT/dy = 0

NEW — Initial Condition:
    T(x, y, t_min) = T_initial   (float or callable)

Changes vs steady PIELM_MixedBC
─────────────────────────────────
1. W gains a 3rd column  (time weights, own scale st = 2√3/t_span)
2. Input X has 3 columns [x, y, t] instead of [x, y]
3. _H_laplacian → _H_pde : adds  - alpha*phi'(z)*W[:,2]  term
4. K_pde = 0  (unchanged — no heat generation)
5. BC points repeated at every time level  (loop over t_e)
   BC H-row types unchanged:
       left   → H_dx      (Neumann flux)
       top    → -k*H_dy - h*H  (Robin convection)
       right  → H_dx      (insulated)
       bottom → H_dy      (insulated)
6. IC rows added at t = t_min  (plain H value rows, raw °C — no normalisation)
"""

import numpy as np
import matplotlib.pyplot as plt


def cheby(n, lo, hi):
    """n Chebyshev interior nodes on [lo, hi]."""
    if n < 1:
        return np.array([])
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))


class PIELM_MixedBC_Transient:
    """
    Transient PIELM with mixed BCs — works directly in °C (no normalisation).
    Convection BC on the top edge anchors the absolute temperature level.

    Solves:  T_xx + T_yy  -  alpha*T_t  =  0
    """

    def __init__(self, hidden_nodes=1800):
        self.N  = hidden_nodes
        self.W  = self.b = self.c = None
        self.k  = None   # stored for flux prediction

    # ── activations ───────────────────────────────────────────────────────────
    def _phi(self, z):      return np.tanh(z)
    def _phi_p(self, z):    return 1.0 - np.tanh(z)**2        # tanh'
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)                            # tanh''

    def _z(self, X):   return X @ self.W.T + self.b.T
    def _H(self, X):   return self._phi(self._z(X))

    def _H_dx(self, X):
        """df/dx rows — same formula as steady, W now has 3 columns."""
        return self._phi_p(self._z(X)) * self.W[:, 0]

    def _H_dy(self, X):
        """df/dy rows — same formula as steady, W now has 3 columns."""
        return self._phi_p(self._z(X)) * self.W[:, 1]

    def _H_pde(self, X, alpha):
        """
        PDE rows for:   T_xx + T_yy  -  alpha*T_t  =  0

        Steady _H_laplacian:   phi''(z) * (W0² + W1²)
        Transient adds:      - alpha * phi'(z) * W2     ← time derivative

        W[:,0]=x-weights, W[:,1]=y-weights, W[:,2]=t-weights (NEW 3rd column)
        Note: _H_dx and _H_dy are unaffected — they only use W[:,0] and W[:,1]
        """
        z      = self._z(X)
        phi_p  = self._phi_p(z)
        phi_pp = self._phi_pp(z)
        H_lap = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)
        H_t   = phi_p  * self.W[:, 2]
        return H_lap - alpha * H_t

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, nx=40, ny=40, nt=15,
              x_max=0.01, y_max=0.01,
              t_min=0.0,  t_max=100.0,
              k=1000., h=25., q_flux=500., T_inf=22.,
              rho=7850., cp=486.,
              T_initial=22.,
              seed=42):
        """
        Parameters
        ----------
        nt        : number of Chebyshev time collocation nodes
        t_min     : start of time domain (set to first ANSYS snapshot if no t=0)
        t_max     : end of time domain
        h         : convection coefficient [W/m²-K]  — top edge
        q_flux    : heat flux into left edge [W/m²]  — left edge
        T_inf     : ambient temperature [°C]          — convection reference
        T_initial : float or callable(x,y) — temperature at t=t_min [°C]
        """
        self.k = k
        alpha  = rho * cp / k

        print(f"[Case 2] alpha={alpha:.3f} s/m²,  steady T_eq≈"
              f"{T_inf + q_flux*x_max/(h*x_max):.1f}°C")

        # random weights — NOW 3 columns (x, y, t)
        sx = 2.0*np.sqrt(3.0) / x_max
        sy = 2.0*np.sqrt(3.0) / y_max
        st = 2.0*np.sqrt(3.0) / max(t_max - t_min, 1e-6)

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
        print(f"[Case 2] Condition number: {best_cond:.2e}")

        # PDE rows — interior (x,y,t) Chebyshev grid, RHS = 0
        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)

        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.zeros((len(X_phys), 1))         # no heat generation

        # BC rows — mixed types, repeated at every time level
        # The H-row type per edge is IDENTICAL to steady:
        #   left   → H_dx         (Neumann)
        #   top    → -k*H_dy-h*H  (Robin)
        #   right  → H_dx         (insulated)
        #   bottom → H_dy         (insulated)
        # Only difference: each set is now repeated for every ti in t_e.
        n_bc = max(nx, ny) * 2
        xe   = np.linspace(0.0, x_max, n_bc)
        ye   = np.linspace(0.0, y_max, n_bc)
        t_e  = np.linspace(t_min, t_max, nt + 2)

        rows_H, rows_K = [], []
        for ti in t_e:

            # 1. LEFT — Neumann flux:  H_dx·c = -q_flux/k
            X_left = np.column_stack((np.zeros(n_bc), ye, np.full(n_bc, ti)))
            rows_H.append(self._H_dx(X_left))
            rows_K.append(np.full((n_bc, 1), -q_flux / k))

            # 2. TOP — Robin convection:  (-k*H_dy - h*H)·c = -h*T_inf
            X_top = np.column_stack((xe, np.full(n_bc, y_max), np.full(n_bc, ti)))
            rows_H.append(-k * self._H_dy(X_top) - h * self._H(X_top))
            rows_K.append(np.full((n_bc, 1), -h * T_inf))

            # 3. RIGHT — insulated:  H_dx·c = 0
            X_right = np.column_stack((np.full(n_bc, x_max), ye, np.full(n_bc, ti)))
            rows_H.append(self._H_dx(X_right))
            rows_K.append(np.zeros((n_bc, 1)))

            # 4. BOTTOM — insulated:  H_dy·c = 0
            X_bot = np.column_stack((xe, np.zeros(n_bc), np.full(n_bc, ti)))
            rows_H.append(self._H_dy(X_bot))
            rows_K.append(np.zeros((n_bc, 1)))

        # IC rows — NEW: spatial domain at t = t_min, plain H value rows, raw °C
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
        K_ic = T_ic_raw.reshape(-1, 1)             # raw °C — no normalisation

        # assemble and solve
        H = np.vstack([H_pde] + rows_H + [H_ic])
        K = np.vstack([K_pde] + rows_K + [K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        n_bc_total = n_bc * 4 * len(t_e)
        print(f"[Case 2] PDE:{len(X_phys)}, BC:{n_bc_total}, IC:{len(X_ic)}, "
              f"N*:{self.N}")
        print(f"[Case 2] System:{H.shape}, rank={rank}, "
              f"smallest SV={sv[-1]:.3e}")
        return self

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, X):
        """X : (n,3) columns [x, y, t].  Returns °C."""
        return (self._H(X) @ self.c).ravel()

    def predict_flux_x(self, X):
        """qx = -k*dT/dx  [W/m²]"""
        return -self.k * (self._H_dx(X) @ self.c).ravel()

    def predict_flux_y(self, X):
        """qy = -k*dT/dy  [W/m²]"""
        return -self.k * (self._H_dy(X) @ self.c).ravel()


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    L = 0.01; k = 45.; h = 25.; q_flux = 500.; T_inf = 22.
    rho = 7850.; cp = 486.

    model = PIELM_MixedBC_Transient(hidden_nodes=5000)
    model.train(nx=22, ny=22, nt=15,
                x_max=L, y_max=L,
                t_min=0.0, t_max=200.0,
                k=k, h=h, q_flux=q_flux, T_inf=T_inf,
                rho=rho, cp=cp,
                T_initial=22.)

    # sanity checks
    # IC at t=0 should be 22
    X_ic = np.column_stack((np.full(5,0.005), np.linspace(0.001,0.009,5), np.zeros(5)))
    print(f"\nIC t=0  (expect 22): {model.predict(X_ic).round(1)}")

    # left flux at t=100 should be ~500 W/m²
    X_l = np.column_stack((np.zeros(5), np.linspace(0.001,0.009,5), np.full(5,100.)))
    print(f"Left flux t=100s (expect ~+500 W/m²): {model.predict_flux_x(X_l).round(1)}")

    # right flux at t=100 should be ~0
    X_r = np.column_stack((np.full(5,L), np.linspace(0.001,0.009,5), np.full(5,100.)))
    print(f"Right flux t=100s (expect ~0):  {model.predict_flux_x(X_r).round(2)}")

    # bottom flux at t=100 should be ~0
    X_b = np.column_stack((np.linspace(0.001,0.009,5), np.zeros(5), np.full(5,100.)))
    print(f"Bottom flux t=100s (expect ~0): {model.predict_flux_y(X_b).round(2)}")

    # at large t, T should approach steady-state 42°C everywhere
    X_ss = np.column_stack((np.full(5,0.005), np.linspace(0.001,0.009,5), np.full(5,180.)))
    print(f"Interior t=180s (expect →42°C): {model.predict(X_ss).round(1)}")

    # time-evolution plot
    xp = np.linspace(0,L,50); yp = np.linspace(0,L,50)
    Xg, Yg = np.meshgrid(xp, yp)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle(f"Case 2 transient  k={k}, h={h}, q_flux={q_flux}, T_inf={T_inf}°C")
    for ax, t_s in zip(axes, [1., 50., 180.]):
        X_ev = np.column_stack((Xg.ravel(), Yg.ravel(), np.full(2500,t_s)))
        T = model.predict(X_ev).reshape(50,50)
        cf = ax.contourf(Xg*1e3, Yg*1e3, T, levels=40, cmap='hot')
        fig.colorbar(cf, ax=ax, label='T (°C)')
        ax.set_title(f't={t_s}s  [{T.min():.1f}–{T.max():.1f}°C]')
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('case2_transient_result.png', dpi=150, bbox_inches='tight')
    print("Saved case2_transient_result.png")