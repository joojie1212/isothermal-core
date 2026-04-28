import numpy as np
from scipy.optimize import fsolve
class Evolve_With_Time():
    def __init__(self,sigma,halo,scanner,t_min=1e-6,t_max=1e3,n_points=1000):
        self.sigma=sigma
        self.halo=halo
        self.scanner=scanner
        self.t_grid = np.linspace(t_min, t_max, n_points)
    def equ_for_r_match(self, r, t):
        return (
            4/np.sqrt(np.pi)
            * self.halo.rho(r)
            * self.halo.sigma_r(r)
            * self.sigma
            - 1/t
        )

    def get_r_grid(self):
        r_grid = []

        for t in self.t_grid:
            r_sol = fsolve(self.equ_for_r_match, 1.0, args=(t,))
            r_grid.append(r_sol[0])
        self.r_grid=np.array(r_grid)
        return self.r_grid
    def evolve(self,scan_kwargs):
        for _r in self.r_grid:
            best,err_map,rho_vals, sigma_vals=self.scanner.scan_rho_sigma(
                   rho_c_range=(self.halo.rho(_r),self.halo.rho(1.e-6)),
                   sigma_range=(0.1*self.halo.v(_r),2*self.halo.v(_r)),
                   r=_r,
                   rho_match=self.halo.rho(_r),m_match=self.halo.m(_r),
                   
                   **scan_kwargs)
