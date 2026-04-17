import matplotlib.pyplot as plt
import numpy as np
from halomodel import NFWhalo
class Plot_Isothermal:
    def __init__(self):
        pass


    def plot_err_map(self,best,err_map, rho_vals, sigma_vals):

        plt.figure(figsize=(7, 5))

    
        log_rho = np.log10(rho_vals)
        log_sigma = np.log10(sigma_vals)

        plt.imshow(
        np.log(err_map.T),
        origin="lower",
        aspect="auto",
        extent=[
            log_rho[0], log_rho[-1],
            log_sigma[0], log_sigma[-1]
        ],
        cmap="viridis"
        )

        plt.colorbar(label="error")


        plt.xlabel(r"$\log_{10}(\rho_c)$")
        plt.ylabel(r"$\log_{10}(\sigma)$")
        plt.title("Boundary Error Map + Local Minima")


        

            

        rho_min = [np.log10(b[1]) for b in best]
        sigma_min = [np.log10(b[2]) for b in best]

        plt.scatter(
                rho_min,
                sigma_min,
                c="red",
                s=60,
                edgecolors="white",
                marker="o",
                label="local minima"
            )

        # annotate
        for i, (e, r, s, _i, j) in enumerate(best):
            plt.text(
                    np.log10(r),
                    np.log10(s),
                    f"{i}",
                    color="white",
                    fontsize=9
                )

        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_nfw_isothermal_match(
        self,
        iso_model,
        params,   # [(rho_c, sigma), ...]
        rho_s, r_s,
        r_match,
        r_min=None,
        r_max=None,
        n_points=1000
        ):

        # -------------------------
        # NFW halo
        # -------------------------
        CDMhalo = NFWhalo(rho_s, r_s)

        # -------------------------
        # 预估范围（用第一个解）
        # -------------------------
        iso_tmp = iso_model()
        iso_tmp.solve()
        iso_tmp.build_dimensionless_interpolators()
        r_iso_tmp, _ = iso_tmp.scaling(params[0][0], params[0][1])

        if r_min is None:
            r_min = min(r_iso_tmp[0], 1e-4 * r_s)

        if r_max is None:
            r_max = max(r_iso_tmp[-1], 10 * r_s)

        r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        rho_nfw_vals = CDMhalo.rho(r)

        mask_inner = r <= r_match
        mask_outer = r >= r_match

        # -------------------------
        # plot
        # -------------------------
        plt.figure(figsize=(7, 5))

        # NFW outer
        plt.loglog(
            r[mask_outer],
            rho_nfw_vals[mask_outer],
            linestyle="--",
            color="black",
            label="NFW (outer)"
        )

        # NFW inner (faded)
        plt.loglog(
            r[mask_inner],
            rho_nfw_vals[mask_inner],
            linestyle="--",
            color="black",
            alpha=0.2
        )

        # -------------------------
        # loop over solutions
        # -------------------------
        for i, (rho_c, sigma) in enumerate(params):

            iso = iso_model()
            iso.solve()
            iso.build_dimensionless_interpolators()

            r_iso, rho_iso = iso.scaling(rho_c, sigma)

            rho_iso_interp = np.interp(
                r,
                r_iso,
                rho_iso,
                left=np.nan,
                right=np.nan
            )

            mask_valid = mask_inner & ~np.isnan(rho_iso_interp)

            plt.loglog(
                r[mask_valid],
                rho_iso_interp[mask_valid],
                alpha=0.7,
                label=f"core {i}" if i < 5 else None  # 防止legend爆炸
            )

        # -------------------------
        # match line
        # -------------------------
        plt.axvline(r_match, color="gray", linestyle=":", alpha=0.6)

        plt.xlabel("r [kpc]")
        plt.ylabel(r"$\rho(r)$ [M$_\odot$/kpc$^3$]")
        plt.title("Multiple Isothermal Cores (Inner Region)")

        plt.legend()
        plt.tight_layout()
        plt.show()