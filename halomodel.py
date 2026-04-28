import numpy as np
from scipy.integrate import quad

class NFWhalo:
    def __init__(self, rho_s, r_s,G=4.302e-6):
        self.rho_s=rho_s
        self.r_s=r_s
        self.G=G
    def rho(self, r):
        x=r/self.r_s
        return self.rho_s/(x*(1+x)**2)
    
    def m(self, r):
        x=r/self.r_s
        ms=4*np.pi*self.rho_s*self.r_s**3
        return ms*(np.log(1+x)-x/(1+x))
    def v2(self, r):
        
        
        integrand = lambda rp: self.rho(rp) * self.G * self.m(rp) / rp**2
        

        r_max = 1e3 * self.r_s
        
        val, _ = quad(integrand, r, r_max, limit=200)
        
        return val / self.rho(r)
    
    def v(self, r):
        return np.sqrt(self.v2(r))
    def omega(self,r,j0):
        s=1.3
        return j0*self.m(r)**s/r**2 