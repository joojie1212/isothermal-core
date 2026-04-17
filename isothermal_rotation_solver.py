import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from isothermal_solver import Isothermal_Solver
class Isothermal_Rotation_Solver(Isothermal_Solver):

    def __init__(self, _omega2, x_min=1e-6, x_max=1e5, n_points=10000):
        super().__init__(x_min, x_max, n_points)   # 调用父类初始化
        self._omega2 = _omega2                         # 新增参数
        
        

    def _ode(self, x, y):
        """
        y = [psi, dpsi/dx]
        """
        psi, dpsi = y

        if x == 0:
            d2psi = 1.0 / 3.0  # series expansion
        else:
            d2psi = -2/x * dpsi + np.exp(-psi+1/3*self._omega2*x**2)

        return [dpsi, d2psi]
    def build_dimensionless_interpolators(self):
        """
        Build interpolators in dimensionless x-space:
        - rho_tilde(x) = exp(-psi)
        - m_tilde(x) = ∫ 4π x^2 exp(-psi) dx
        """

        # ---------- dimensionless density ----------
        self.rho_tilde = np.exp(-self.psi+1/3*self._omega2*self.x**2)

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

        return self.rho_tilde_interp, self.m_tilde_interp