import numpy as np
from isothermal_solver import Isothermal_Solver
from isothermal_scan import Isothermal_Scanner
from isothermal_rotation_solver import Isothermal_Rotation_Solver
from isothermal_rotation_scan import Isothermal_Rotation_Scanner
from halomodel import NFWhalo
from plot import Plot_Isothermal
import matplotlib.pyplot as plt
scanner=Isothermal_Scanner(Isothermal_Solver)
scanner2=Isothermal_Rotation_Scanner(Isothermal_Rotation_Solver,100000)
CDMhalo=NFWhalo(1.94e7,2.586)
plot_isothermal=Plot_Isothermal()

best,err_map,rho_vals, sigma_vals=scanner.scan_rho_sigma(
                   rho_c_range=(CDMhalo.rho(10),CDMhalo.rho(0.00001)),
                   sigma_range=(0.05,100),
                   r=0.001,
                   rho_match=CDMhalo.rho(0.001),m_match=CDMhalo.m(0.001),
                   
                   N_rho=100,
                   N_sigma=100,
                   smooth_sigma=1.0,
                   n_peaks=1)
#plot_isothermal.plot_err_map(best,err_map,rho_vals=(CDMhalo.rho(10),CDMhalo.rho(0.0000001)),sigma_vals=(0.05,100))
best2,err_map,rho_vals, sigma_vals=scanner2.scan_rho_sigma(
                   rho_c_range=(CDMhalo.rho(10),CDMhalo.rho(0.00001)),
                   sigma_range=(0.05,100),
                   r=0.001,
                   rho_match=CDMhalo.rho(0.001),m_match=CDMhalo.m(0.001),
                   
                   N_rho=100,
                   N_sigma=100,
                   smooth_sigma=1.0,
                   n_peaks=1)

plot_isothermal.plot_nfw_isothermal_match(
    Isothermal_Solver,
    [(best[0][1],best[0][2]),(best2[0][1], best2[0][2])],
    1.94e7,2.586,
    0.001,
    r_min=None,
    r_max=None,
    n_points=1000
)