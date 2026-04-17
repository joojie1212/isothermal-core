import numpy as np
from isothermal_solver import Isothermal_Solver
from isothermal_scan import Isothermal_Scanner
class Isothermal_Rotation_Scanner(Isothermal_Scanner):
    def __init__(self,solver,omega2,G=4.302e-6,x_min=1e-6,x_max=1e5,n_points=1000):
        """
        solver_cls: IsothermalSphere class (not instance)
        """
        self.x_min=x_min
        self.x_max=x_max
        self.n_points=n_points
        self.G=G
        self.solver = solver
        self.scanner=Isothermal_Scanner(Isothermal_Solver)
        self.omega2=omega2
        
    def scan_rho_sigma(self,
                   rho_c_range,
                   sigma_range,
                   r,
                   rho_match,m_match,err_rho_c=0.3,
                   
                   N_rho=50,
                   N_sigma=50,
                   smooth_sigma=1.0,
                   n_peaks=5):
        if not hasattr(self, "_omega2"):
  

            best,err_map,rho_vals, sigma_vals=self.scanner.scan_rho_sigma(
                   rho_c_range,
                   sigma_range,
                   r,
                   rho_match,m_match,
                   
                   N_rho,
                   N_sigma,
                   smooth_sigma,
                   n_peaks=1)
            self.rho_c=best[0][1]
            self._omega2=self.omega2/(4*np.pi*self.G*self.rho_c)
        self.model = self.solver(self._omega2,x_min=self.x_min,x_max=self.x_max,n_points=self.n_points)
        self.model.solve()
        self.model.build_dimensionless_interpolators()
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
        if abs(self.best[0][1]-self.rho_c)/self.rho_c<err_rho_c :

            return self.best, self.err_map, rho_vals, sigma_vals
        else:
            
            self.rho_c=self.best[0][1]
            self._omega2=self.omega2/(4*np.pi*self.G*self.rho_c)
            return self.scan_rho_sigma(rho_c_range,sigma_range,r,rho_match,m_match,err_rho_c,N_rho,N_sigma,smooth_sigma,n_peaks)

    