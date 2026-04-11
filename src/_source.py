import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chebyshev interior collocation, clusters near boundaries
def cheby(n, lo, hi):
    k = np.arange(1, n + 1)
    return np.sort(0.5*(lo+hi) + 0.5*(hi-lo)*np.cos(np.pi*(2*k-1)/(2*n)))

#========================================================================================#

""" Case 1: Steady diffusion """
""" u_xx + u_yy = 0 """ 
class PIELM2DDiffusion:
    def __init__(self, hidden_nodes=200):
        self.hidden_nodes = hidden_nodes
        self.W = None
        self.b = None
        self.c = None
        self._T_ref = None
        self._T_scale = None

    def _phi(self, z):
        return np.tanh(z)

    def _phi_double_prime(self, z):
        t = np.tanh(z)
        return -2.0 * t * (1.0 - t ** 2)

    def _z(self, X):
        return X @ self.W.T + self.b.T

    def _H(self, X):
        return self._phi(self._z(X))

    def _H_laplacian(self, X):
        z = self._z(X)
        phi_pp = self._phi_double_prime(z)
        return phi_pp * (self.W[:, 0] ** 2 + self.W[:, 1] ** 2)

    def train(self, nx=40, ny=40, x_max=0.01, y_max=0.01,T_left=700.0, T_other=500.0, seed=42):
        self._T_ref = T_other
        self._T_scale = T_left - T_other

        # Weight scale: W·x should span ~O(1) so tanh stays curved
        # For uniform [-s,s]: set s = 2*sqrt(3)/L
        scale_x = 2.0 * np.sqrt(3.0) / x_max
        scale_y = 2.0 * np.sqrt(3.0) / y_max

        # Try 5 seeds, keep best-conditioned hidden matrix
        best_cond = np.inf
        best_W, best_b = None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_try = np.column_stack([np.random.uniform(-scale_x, scale_x, self.hidden_nodes),
                np.random.uniform(-scale_y, scale_y, self.hidden_nodes),
            ])
            b_try = np.random.uniform(-1.0, 1.0, (self.hidden_nodes, 1))
            X_sample = np.random.rand(100, 2) * [x_max, y_max]
            H_s = np.tanh(X_sample @ W_try.T + b_try.T)
            cond = np.linalg.cond(H_s)
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_try, b_try

        self.W, self.b = best_W, best_b
        logger.info(f"Best condition number (sample H): {best_cond}")

        xc = cheby(nx - 2, 0.0, x_max)
        yc = cheby(ny - 2, 0.0, y_max)
        Xg, Yg = np.meshgrid(xc, yc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel()))

        H_pde = self._H_laplacian(X_phys)
        K_pde = np.zeros((len(X_phys), 1))
        n_bc = max(nx, ny) * 2
        x_e = np.linspace(0.0, x_max, n_bc)
        y_e = np.linspace(0.0, y_max, n_bc)

        X_bc, K_bc = [], []
        for yi in y_e:
            X_bc.append([0.0, yi]); K_bc.append([1.0])   # left
        for yi in y_e:
            X_bc.append([x_max, yi]); K_bc.append([0.0])   # right
        for xi in x_e[1:-1]:
            X_bc.append([xi, 0.0]); K_bc.append([0.0])   # bottom
        for xi in x_e[1:-1]:
            X_bc.append([xi, y_max]); K_bc.append([0.0])   # top

        X_bc = np.array(X_bc)
        K_bc = np.array(K_bc)

        H = np.vstack([H_pde, self._H(X_bc)])
        K = np.vstack([K_pde, K_bc])

        # self.c, _ , _ = np.linalg.lstsq(H, K, rcond=None)
        self.c = np.linalg.inv(H.T @ H) @ H.T @ K 
        return self

    def predict(self, X, return_celsius=True):
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref if return_celsius else u
    
#========================================================================================#
    
""" Case 2: Steady diffusion with internal heat generation """
"""u_xx + u_yy = S  (S = -q/k, a constant source term)"""
class PIELM_HeatGen:
    def __init__(self, hidden_nodes=1800):
        self.N = hidden_nodes
        self.W = self.b = self.c = None
        self._T_ref = self._T_scale = None

    def _phi(self, z):
        return np.tanh(z)
    
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2*t*(1 - t**2)

    def _z(self, X):   
        return X @ self.W.T + self.b.T
    
    def _H(self, X):   
        return self._phi(self._z(X))

    def _H_laplacian(self, X):
        z = self._z(X)
        return self._phi_pp(z) * (self.W[:, 0]**2 + self.W[:, 1]**2)
    
    def train(self, nx=40, ny=40, x_max=0.01, y_max=0.01, T_left=700., T_other=500., k=50., q=1e9, T_scale=200, seed=42):
        self._T_ref   = T_other
        self._T_scale = T_scale

        # Source term normalised
        S_norm = (-q / k) / self._T_scale

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
        logger.info(f" Hidden layer condition number: {best_cond}")

        # interior Chebyshev grid 
        xc = cheby(nx-2, 0., x_max)
        yc = cheby(ny-2, 0., y_max)
        Xg, Yg = np.meshgrid(xc, yc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel()))

        H_pde = self._H_laplacian(X_phys)
        K_pde = np.full((len(X_phys), 1), S_norm)

        n_bc = max(nx, ny) * 2
        xe = np.linspace(0., x_max, n_bc)
        ye = np.linspace(0., y_max, n_bc)

        X_bc, K_bc = [], []

        for yi in ye:
            X_bc.append([0., yi]);
            K_bc.append([1.])  # left
        for yi in ye:
            X_bc.append([x_max, yi]); 
            K_bc.append([0.])  # right
        for xi in xe[1:-1]:
            X_bc.append([xi, 0.]);
            K_bc.append([0.])  # bot
        for xi in xe[1:-1]: 
            X_bc.append([xi, y_max]);
            K_bc.append([0.])  # top

        X_bc = np.array(X_bc);  K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        H = np.vstack([H_pde, H_bc])
        K = np.vstack([K_pde, K_bc])
        # self.c = np.linalg.inv(H.T @ H) @ H.T @ K
        self.c, _, _, _ = np.linalg.lstsq(H, K, rcond=None)
        return self

    # prediction 
    def predict(self, X):
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref

#========================================================================================#

""" Case 3: Mixed BCs with convection and flux """
class PIELM_MixedBC:

    def __init__(self, hidden_nodes=1800):
        self.N = hidden_nodes
        self.W = self.b = self.c = None

    def _phi(self, z):
        return np.tanh(z)
    def _phi_p(self, z):
        return 1 - np.tanh(z)**2
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2*t*(1 - t**2)  

    def _z(self, X):   
        return X @ self.W.T + self.b.T

    def _H(self, X):
        return self._phi(self._z(X))

    def _H_dx(self, X):
        return self._phi_p(self._z(X)) * self.W[:, 0]

    def _H_dy(self, X):
        return self._phi_p(self._z(X)) * self.W[:, 1]

    def _H_laplacian(self, X):
        z = self._z(X)
        return self._phi_pp(z) * (self.W[:, 0]**2 + self.W[:, 1]**2)
    
    def train(self, nx=40, ny=40, x_max=0.01, y_max=0.01, k=50., h=25., q_flux=500., T_inf=22., seed=42):
        self.k = k;  self.h = h
        self.q_flux = q_flux;  self.T_inf = T_inf

        sx = 2*np.sqrt(3) / x_max
        sy = 2*np.sqrt(3) / y_max

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([np.random.uniform(-sx, sx, self.N),
                np.random.uniform(-sy, sy, self.N),
            ])
            b_ = np.random.uniform(-1, 1, (self.N, 1))
            Xs = np.random.rand(100, 2) * [x_max, y_max]
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        logger.info(f"[Case 2] Hidden layer condition number: {best_cond}")

        xc = cheby(nx-2, 0., x_max)
        yc = cheby(ny-2, 0., y_max)
        Xg, Yg = np.meshgrid(xc, yc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel()))

        H_pde = self._H_laplacian(X_phys)
        K_pde = np.zeros((len(X_phys), 1))       # Laplace: RHS = 0

        n_bc = max(nx, ny) * 2
        xe = np.linspace(0., x_max, n_bc)
        ye = np.linspace(0., y_max, n_bc)

        rows_H, rows_K = [], []

        # 1. LEFT EDGE (x=0) 
        # -k · du/dx = q_flux  =>  du/dx = -q_flux/k
        X_left = np.column_stack((np.zeros(n_bc), ye))
        rhs_left = -q_flux / k              
        rows_H.append(self._H_dx(X_left))
        rows_K.append(np.full((n_bc, 1), rhs_left))

        # 2. TOP EDGE (y=L) 
        # -k·dT/dy - h·T = -h·T_inf   
        X_top = np.column_stack((xe, np.full(n_bc, y_max)))
        H_top_conv = -k * self._H_dy(X_top) - h * self._H(X_top)
        rhs_top = -h * T_inf
        rows_H.append(H_top_conv)
        rows_K.append(np.full((n_bc, 1), rhs_top))

        # 3. RIGHT EDGE (x=L) 
        # insulated  du/dx = 0
        X_right = np.column_stack((np.full(n_bc, x_max), ye))
        rows_H.append(self._H_dx(X_right))
        rows_K.append(np.zeros((n_bc, 1)))

        # 4. BOTTOM EDGE (y=0) 
        # insulated  du/dy = 0
        X_bot = np.column_stack((xe, np.zeros(n_bc)))
        rows_H.append(self._H_dy(X_bot))
        rows_K.append(np.zeros((n_bc, 1)))

        H = np.vstack([H_pde] + rows_H)
        K = np.vstack([K_pde] + rows_K)

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        return self

    def predict(self, X):
        return (self._H(X) @ self.c).ravel()

    # def predict_flux_x(self, X):
    #     return -self.k * (self._H_dx(X) @ self.c).ravel()

    # def predict_flux_y(self, X):
    #     return -self.k * (self._H_dy(X) @ self.c).ravel()
    
#========================================================================================#

""" Case 4: Transient diffusion with internal heat generation """
"""u_xx + u_yy  -  alpha * u_t  =  0"""
class PIELM2DDiffusion_transient:

    def __init__(self, hidden_nodes=2000):
        self.hidden_nodes = hidden_nodes
        self.W = self.b = self.c = None
        self._T_ref = self._T_scale = None

    def _phi(self, z):      
        return np.tanh(z)
    
    def _phi_p(self, z): 
        return 1.0 - np.tanh(z)**2
    
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)

    def _z(self, X):
        return X @ self.W.T + self.b.T
    
    def _H(self, X):
        return self._phi(self._z(X))

    def _H_pde(self, X, alpha):
        z = self._z(X)
        phi_p = self._phi_p(z)                  
        phi_pp = self._phi_pp(z)                 
        H_lap = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)
        H_t = phi_p  * self.W[:, 2]           
        return H_lap - alpha * H_t

    def train(self, nx=20, ny=20, nt=10, x_max=0.01, y_max=0.01, t_min=0.0, t_max=100.0, rho=7850.0, cp=434.0, k=50.0,
              T_scale = 200.0, T_left=700.0, T_other=500.0, T_initial=500.0, seed=42):

        self._T_ref   = T_other
        self._T_scale = T_scale
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
        logger.info(f"Best condition number: {best_cond}")

        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)
        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.zeros((len(X_phys), 1))

        n_bc = max(nx, ny) * 2
        x_e = np.linspace(0.0, x_max, n_bc)
        y_e = np.linspace(0.0, y_max, n_bc)
        t_e = np.linspace(t_min, t_max, nt + 2)

        X_bc, K_bc = [], []
        for ti in t_e:
            for yi in y_e:
                X_bc.append([0.0, yi, ti]);  K_bc.append([normalise(T_left)])
            for yi in y_e:
                X_bc.append([x_max, yi, ti]);  K_bc.append([normalise(T_other)])
            for xi in x_e[1:-1]:
                X_bc.append([xi, 0.0, ti]);  K_bc.append([normalise(T_other)])
            for xi in x_e[1:-1]:
                X_bc.append([xi, y_max, ti]);  K_bc.append([normalise(T_other)])

        X_bc = np.array(X_bc); K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        xc_ic = np.linspace(0.0, x_max, nx)
        yc_ic = np.linspace(0.0, y_max, ny)
        Xg_ic, Yg_ic = np.meshgrid(xc_ic, yc_ic)
        X_ic = np.column_stack((Xg_ic.ravel(), Yg_ic.ravel(), np.full(Xg_ic.size, t_min)))  
        H_ic = self._H(X_ic)

        T_ic_raw = np.full(len(X_ic), float(T_initial))
        K_ic = normalise(T_ic_raw).reshape(-1, 1)

        H = np.vstack([H_pde, H_bc, H_ic])
        K = np.vstack([K_pde, K_bc, K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        return self

    # prediction 
    def predict(self, X, return_celsius=True):
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref if return_celsius else u
    
#========================================================================================#

""" Case 5: Transient diffusion with internal heat generation """
"""  u_xx + u_yy  -  alpha*u_t  =  S_norm
    where    alpha  = rho*cp / k
             S_norm = -q / (k * T_scale)
   [non-zero, same sign as steady]
    """
class PIELM_HeatGen_Transient:

    def __init__(self, hidden_nodes=1800):
        self.N  = hidden_nodes
        self.W  = self.b = self.c = None
        self._T_ref = self._T_scale = None

    def _phi(self, z):
        return np.tanh(z)
    
    def _phi_p(self, z):
        return 1.0 - np.tanh(z)**2
    
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)

    def _z(self, X):
        return X @ self.W.T + self.b.T
    
    def _H(self, X):
        return self._phi(self._z(X))

    def _H_pde(self, X, alpha):
        z = self._z(X)
        phi_p = self._phi_p(z)
        phi_pp = self._phi_pp(z)
        H_lap = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)  # Laplacian rows
        H_t = phi_p  * self.W[:, 2] # time deriv rows
        return H_lap - alpha * H_t
 
    def train(self, nx=40, ny=40, nt=15, x_max=0.01, y_max=0.01, t_min=0.0,  t_max=100.0, T_left=700., T_other=500.,
              k=50., q=1e9, rho=7850., cp=434., T_scale = 200., T_initial=500., seed=42):
        
        self._T_ref   = T_other
        self._T_scale = T_scale

        def norm(T):
            return (T - self._T_ref) / self._T_scale

        alpha = rho * cp / k
        S_norm = (-q / k) / self._T_scale   # same as steady, non-zero

        logger.info(f"alpha={alpha} s/m^2,  S_norm={S_norm}")

        sx = 2.0*np.sqrt(3.0) / x_max
        sy = 2.0*np.sqrt(3.0) / y_max
        st = 2.0*np.sqrt(3.0) / max(t_max - t_min, 1e-6)  # own time scale

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([np.random.uniform(-sx, sx, self.N),
                np.random.uniform(-sy, sy, self.N),
                np.random.uniform(-st, st, self.N),   
            ])
            b_ = np.random.uniform(-1.0, 1.0, (self.N, 1))
            Xs = np.random.rand(100, 3)
            Xs[:,0] *= x_max
            Xs[:,1] *= y_max
            Xs[:,2] = Xs[:,2]*(t_max - t_min) + t_min
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        logger.info(f"Condition number: {best_cond}")

        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)

        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.full((len(X_phys), 1), S_norm) 

        n_bc = max(nx, ny) * 2
        xe = np.linspace(0.0, x_max, n_bc)
        ye = np.linspace(0.0, y_max, n_bc)
        t_e = np.linspace(t_min, t_max, nt + 2)

        X_bc, K_bc = [], []
        for ti in t_e:
            for yi in ye:
                X_bc.append([0.0, yi, ti]);  K_bc.append([norm(T_left)])
            for yi in ye:
                X_bc.append([x_max, yi, ti]);  K_bc.append([norm(T_other)])
            for xi in xe[1:-1]:
                X_bc.append([xi, 0.0, ti]);  K_bc.append([norm(T_other)])
            for xi in xe[1:-1]:
                X_bc.append([xi, y_max, ti]);  K_bc.append([norm(T_other)])

        X_bc = np.array(X_bc);  K_bc = np.array(K_bc)
        H_bc = self._H(X_bc)

        # IC's
        xc_ic = np.linspace(0.0, x_max, nx)
        yc_ic = np.linspace(0.0, y_max, ny)
        Xg_ic, Yg_ic = np.meshgrid(xc_ic, yc_ic)
        X_ic = np.column_stack((Xg_ic.ravel(), Yg_ic.ravel(), np.full(Xg_ic.size, t_min)))
        H_ic = self._H(X_ic)

        T_ic_raw = np.full(len(X_ic), float(T_initial))
        K_ic = norm(T_ic_raw).reshape(-1, 1)

        H = np.vstack([H_pde, H_bc, H_ic])
        K = np.vstack([K_pde, K_bc, K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        
        return self

    def predict(self, X):
        u = (self._H(X) @ self.c).ravel()
        return u * self._T_scale + self._T_ref

#========================================================================================#

""" Case 6: Transient diffusion with mixed BCs (convection + flux) """
class PIELM_MixedBC_Transient:

    def __init__(self, hidden_nodes=1800):
        self.N  = hidden_nodes
        self.W  = self.b = self.c = None
        self.k  = None   

    def _phi(self, z):
        return np.tanh(z)
    
    def _phi_p(self, z):
        return 1.0 - np.tanh(z)**2
    
    def _phi_pp(self, z):
        t = np.tanh(z)
        return -2.0*t*(1.0 - t**2)

    def _z(self, X):   
        return X @ self.W.T + self.b.T
    
    def _H(self, X):
        return self._phi(self._z(X))

    def _H_dx(self, X):
        return self._phi_p(self._z(X)) * self.W[:, 0]

    def _H_dy(self, X):
        return self._phi_p(self._z(X)) * self.W[:, 1]

    def _H_pde(self, X, alpha):
        z = self._z(X)
        phi_p = self._phi_p(z)
        phi_pp = self._phi_pp(z)
        H_lap = phi_pp * (self.W[:, 0]**2 + self.W[:, 1]**2)
        H_t = phi_p  * self.W[:, 2]
        return H_lap - alpha * H_t

    def train(self, nx=40, ny=40, nt=15, x_max=0.01, y_max=0.01, t_min=0.0,  t_max=100.0, k=50., h=25., q_flux=500., T_inf=22.,
              rho=7850., cp=434., T_initial=22., seed=42):

        self.k = k
        alpha  = rho * cp / k

        sx = 2.0*np.sqrt(3.0) / x_max
        sy = 2.0*np.sqrt(3.0) / y_max
        st = 2.0*np.sqrt(3.0) / max(t_max - t_min, 1e-6)

        best_cond, best_W, best_b = np.inf, None, None
        for s in range(seed, seed + 5):
            np.random.seed(s)
            W_ = np.column_stack([np.random.uniform(-sx, sx, self.N),
                np.random.uniform(-sy, sy, self.N),
                np.random.uniform(-st, st, self.N),
            ])
            b_ = np.random.uniform(-1.0, 1.0, (self.N, 1))
            Xs = np.random.rand(100, 3)
            Xs[:,0] *= x_max
            Xs[:,1] *= y_max
            Xs[:,2] = Xs[:,2]*(t_max - t_min) + t_min
            cond = np.linalg.cond(np.tanh(Xs @ W_.T + b_.T))
            if cond < best_cond:
                best_cond, best_W, best_b = cond, W_, b_
        self.W, self.b = best_W, best_b
        logger.info(f"Condition number: {best_cond}")

        xc = cheby(nx, 0.0, x_max)
        yc = cheby(ny, 0.0, y_max)
        tc = cheby(nt, t_min, t_max)

        Xg, Yg, Tg = np.meshgrid(xc, yc, tc)
        X_phys = np.column_stack((Xg.ravel(), Yg.ravel(), Tg.ravel()))

        H_pde = self._H_pde(X_phys, alpha)
        K_pde = np.zeros((len(X_phys), 1))        
        
        n_bc = max(nx, ny) * 2
        xe = np.linspace(0.0, x_max, n_bc)
        ye = np.linspace(0.0, y_max, n_bc)
        t_e = np.linspace(t_min, t_max, nt + 2)

        rows_H, rows_K = [], []
        for ti in t_e:
            # 1. LEFT 
            X_left = np.column_stack((np.zeros(n_bc), ye, np.full(n_bc, ti)))
            rows_H.append(self._H_dx(X_left))
            rows_K.append(np.full((n_bc, 1), -q_flux / k))

            # 2. TOP 
            X_top = np.column_stack((xe, np.full(n_bc, y_max), np.full(n_bc, ti)))
            rows_H.append(-k * self._H_dy(X_top) - h * self._H(X_top))
            rows_K.append(np.full((n_bc, 1), -h * T_inf))

            # 3. RIGHT 
            X_right = np.column_stack((np.full(n_bc, x_max), ye, np.full(n_bc, ti)))
            rows_H.append(self._H_dx(X_right))
            rows_K.append(np.zeros((n_bc, 1)))

            # 4. BOTTOM
            X_bot = np.column_stack((xe, np.zeros(n_bc), np.full(n_bc, ti)))
            rows_H.append(self._H_dy(X_bot))
            rows_K.append(np.zeros((n_bc, 1)))

        # IC rows 
        xc_ic = np.linspace(0.0, x_max, nx)
        yc_ic = np.linspace(0.0, y_max, ny)
        Xg_ic, Yg_ic = np.meshgrid(xc_ic, yc_ic)
        X_ic = np.column_stack((Xg_ic.ravel(), Yg_ic.ravel(),np.full(Xg_ic.size, t_min)))
        H_ic = self._H(X_ic)

        T_ic_raw = np.full(len(X_ic), float(T_initial))
        K_ic = T_ic_raw.reshape(-1, 1)
        
        H = np.vstack([H_pde] + rows_H + [H_ic])
        K = np.vstack([K_pde] + rows_K + [K_ic])

        self.c, _, rank, sv = np.linalg.lstsq(H, K, rcond=None)
        return self
 
    def predict(self, X):
        return (self._H(X) @ self.c).ravel()

 
