import numpy as np
from scipy.integrate import quad
class NFWhalo:
    def __init__(self, rho_s, r_s):
        self.rho_s=rho_s
        self.r_s=r_s
        
    def rho(self, r):
        x=r/self.r_s
        return self.rho_s/(x*(1+x)**2)
    
    def m(self, r):
        x=r/self.r_s
        ms=4*np.pi*self.rho_s*self.r_s**3
        return ms*(np.log(1+x)-x/(1+x))
    def sigma_r2(self, r):
        
        
        integrand = lambda rp: self.rho(rp) * self.G * self.m(rp) / rp**2
        
        # 上限要截断（∞ → r_max）
        r_max = 1e3 * self.r_s
        
        val, _ = quad(integrand, r, r_max, limit=200)
        
        return val / self.rho(r)
    
    def sigma_r(self, r):
        return np.sqrt(self.sigma_r2(r))
 