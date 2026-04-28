import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import minimum_filter
class Isothermal_Scanner:
    def __init__(self,solver,x_min=1e-6,x_max=1e10,n_points=100000):
        """
        solver_cls: IsothermalSphere class (not instance)
        """
        self.solver = solver
        self.model = self.solver(x_min=x_min,x_max=x_max,n_points=n_points)
        self.model.solve()
        self.model.build_dimensionless_interpolators()

    def re_solver(self):
        """
        to prevent re calculation
        """
        return self.model

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
    
    def find_global_minima(self, err_map, rho_vals, sigma_vals, n_peaks=5, neighborhood=5):
        """
        使用局部极小值检测寻找多个 minima（包括浅局部最小值）

        Parameters
        ----------
        err_map : 2D ndarray
            误差矩阵
        rho_vals : 1D ndarray
            行对应 rho
        sigma_vals : 1D ndarray
            列对应 sigma
        n_peaks : int
            返回前几个最优局部极小值
        neighborhood : int
            邻域窗口大小（建议奇数：3,5,7）

        Returns
        -------
        candidates : list of tuple
            (error, rho, sigma, i, j)
        """

        import numpy as np
        from scipy.ndimage import minimum_filter

        err = err_map.copy()
        rows, cols = err.shape

        # 找局部最小值：该点等于邻域最小值
        local_min_mask = (err == minimum_filter(err, size=neighborhood, mode='nearest'))

        # 提取所有局部最小点坐标
        indices = np.argwhere(local_min_mask)

        candidates = []
        for i, j in indices:
            candidates.append(
                (err[i, j], rho_vals[i], sigma_vals[j], i, j)
            )

        # 按误差从小到大排序
        candidates.sort(key=lambda x: x[0])

        # 取前 n_peaks
        candidates = candidates[:n_peaks]

        # 检查边界极值点（只警告，不中断）
        for k, (val, rho, sigma, i, j) in enumerate(candidates, start=1):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                print(
                    f"Warning: Candidate #{k} is on boundary "
                    f"(i={i}, j={j}, rho={rho}, sigma={sigma}, err={val:.6g}). "
                    f"Consider expanding search range."
                )

        return candidates
        