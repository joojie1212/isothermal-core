import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import minimum_filter
class Isothermal_Scanner:
    def __init__(self,solver,x_min=1e-6,x_max=1e5,n_points=1000):
        """
        solver_cls: IsothermalSphere class (not instance)
        """
        self.solver = solver
        self.model = self.solver(x_min=x_min,x_max=x_max,n_points=n_points)
        self.model.solve()
        self.model.build_dimensionless_interpolators()



    def boundary_error(self, rho_c, sigma, r, rho_match, m_match):

        self.model.scaling(rho_c, sigma)
        rho_edge, m_edge = self.model.query(r)

        err_rho = (rho_edge - rho_match) / rho_match
        err_m   = (m_edge - m_match) / m_match

        return err_m**2 +err_rho**2
    
    


    def scan_rho_sigma(self,
                   rho_c_range,
                   sigma_range,
                   r,
                   rho_match,m_match,
                   
                   N_rho=50,
                   N_sigma=50,
                   smooth_sigma=1.0,
                   n_peaks=5):
 

        rho_vals = np.logspace(np.log10(rho_c_range[0]),
                           np.log10(rho_c_range[1]),
                           N_rho)

        sigma_vals = np.logspace(np.log10(sigma_range[0]),
                             np.log10(sigma_range[1]),
                             N_sigma)

        self.err_map = np.zeros((N_rho, N_sigma))

    
        for i, rho_c in enumerate(rho_vals):
            for j, sigma in enumerate(sigma_vals):

                

                self.err_map[i, j] = self.boundary_error(
                rho_c, sigma, r,
                rho_match=rho_match,
                m_match=m_match
                )


        err_smooth = self.err_map#gaussian_filter(self.err_map, smooth_sigma)


        candidates = self.find_global_minima(err_smooth, rho_vals, sigma_vals)


        candidates.sort(key=lambda x: x[0])

        self.best = candidates[:n_peaks]

        return self.best, self.err_map, rho_vals, sigma_vals
    
    def find_global_minima(self, err_map, rho_vals, sigma_vals, n_peaks=5):

        err = err_map.copy()
        candidates = []

        for _ in range(n_peaks):

            idx = np.argmin(err)
            i, j = np.unravel_index(idx, err.shape)

            candidates.append((err[i,j], rho_vals[i], sigma_vals[j], i, j))

        # suppress neighborhood
            r = 3
            err[max(0,i-r):i+r+1, max(0,j-r):j+r+1] = np.inf

        return candidates
    
    