import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

class Isothermal_Solver:
    def __init__(self,x_min=1e-6,x_max=1e3,n_points=1000):
        self.x_min=x_min
        self.x_max=x_max
        self.n_points=n_points
        self.logx_min = np.log10(self.x_min)
        self.logx_max = np.log10(self.x_max)
        self.dlogx = (self.logx_max - self.logx_min) / self.n_points
        
        
    def _ode(self, x, y):
        """
        y = [psi, dpsi/dx]
        """
        psi, dpsi = y

        if x == 0:
            d2psi = 1.0 / 3.0  # series expansion
        else:
            d2psi = -2/x * dpsi + np.exp(-psi)

        return [dpsi, d2psi]
   
    def solve(self):

        x_span = (self.x_min, self.x_max)

        y0 = [self.x_min**2 / 6, self.x_min / 3]

        sol = solve_ivp(
        self._ode,
        x_span,
        y0,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
        )

        self.x = np.logspace(
        self.logx_min,
        self.logx_max,
        self.n_points
    )

        psi = sol.sol(self.x)

        self.psi = psi[0]
        self.dpsi = psi[1]

        return self.x, self.psi, self.dpsi


    def update_grid(self, x_new):

        self.x_max = x_new
        self.logx_max = np.log10(x_new)

        new_n_points = int((self.logx_max - self.logx_min) / self.dlogx)+1

        self.n_points = max(new_n_points, self.n_points)
    def build_dimensionless_interpolators(self):
        """
        Build interpolators in dimensionless x-space:
        - rho_tilde(x) = exp(-psi)
        - m_tilde(x) = ∫ 4π x^2 exp(-psi) dx
        """

        # ---------- dimensionless density ----------
        self.rho_tilde = np.exp(-self.psi)

        # ---------- mass integration in x-space ----------
        dx = np.diff(self.x)

        integrand = 4 * np.pi * self.x**2 * self.rho_tilde

        m = np.zeros_like(self.x)
        m = cumulative_trapezoid(
        self.x**2*self.rho_tilde,
        self.x,
        initial=0
        )

        self.m_tilde = m
        I = np.zeros_like(self.x)
        I = cumulative_trapezoid(
        self.x**4*self.rho_tilde,
        self.x,
        initial=0
        )

        self.I_tilde = I
        # ---------- interpolators ----------
        self.rho_tilde_interp = interp1d(
            np.log(self.x),
            np.log(self.rho_tilde),
            bounds_error=False,
            fill_value=np.nan
        )

        self.m_tilde_interp = interp1d(
            self.x,
            self.m_tilde,
            bounds_error=False,
            fill_value=np.nan
        )
        self.I_tilde_interp = interp1d(
            self.x,
            self.I_tilde,
            bounds_error=False,
            fill_value=np.nan
        )

        return self.rho_tilde_interp, self.m_tilde_interp, self.I_tilde_interp
    def scaling(self, rho_c, sigma, G=4.302e-6):
        """
        rho_c : central density (Msun/kpc^3)
        sigma : velocity dispersion (km/s)
        G     : gravitational constant (default in kpc, Msun, km/s)
        """
        if not hasattr(self, "psi"):
            raise RuntimeError("Call solve() before scaling()")
        self.rho_c = rho_c
        self.sigma = sigma
        self.G = G

        # scale radius
        self.r0 = np.sqrt(sigma**2 / (4 * np.pi * G * rho_c))
        # density profile
        self.r = self.x * self.r0
        self.rho = self.rho_c * self.rho_tilde
        self.m_scale=4*np.pi*self.r0**3*self.rho_c
        self.I_scale=4*np.pi*self.r0**5*self.rho_c
        
        

        return self.r, self.rho

    def query(self, r_query):
        """
        Return rho(r) and M(r) at given physical radius r_query.
        Automatically extends solution if needed.
        """
        if not hasattr(self, "r"):
            raise RuntimeError("Call scaling() before query()")

        

        r_max = self.r[-1]

        if r_query > r_max:

            self.update_grid(max(r_query*1.2,r_max*2))
            print("----warning! large core radius! resolving isothermal core----")
            self.solve()
            self.build_dimensionless_interpolators()
            self.scaling(self.rho_c, self.sigma, self.G)
        
       
        rho_q = np.exp(self.rho_tilde_interp(np.log(r_query/self.r0)))*self.rho_c
        m_q = self.m_tilde_interp(r_query/self.r0)*self.m_scale

        return rho_q, m_q

    def rotational_inertia(self,r):
        if not hasattr(self, "r"):
            raise RuntimeError("Call scaling() before query()")

        

        r_max = self.r[-1]

        if r > r_max:

            self.update_grid(max(r*1.2,r_max*2))
            print("----warning! large core radius! resolving isothermal core----")
            self.solve()
            self.build_dimensionless_interpolators()
            self.scaling(self.rho_c, self.sigma, self.G)
        
       

        I_q = self.I_tilde_interp(r/self.r0)*self.I_scale
        return I_q

    def potential(self):
        """
        Return physical potential Phi = sigma^2 * psi
        """
        return self.sigma**2 * self.psi