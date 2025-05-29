import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from poppy import zernike
from scipy.interpolate import griddata

from rfcml_bending.force_matrix import ForceSpace

TEST_OUTPUT_DATA_DIR = Path(__file__).parents[2].joinpath("tests")


class Bending:
    def __init__(self, filename):
        self.force_space = ForceSpace(filename)

    # bending mode fit function
    def fit_bending(self, modes_to_fit, map):
        """
        Fits bending modes.
        Requires the data be sampled very specifically so as to match
        the FEA nodes in the force_space matrix.
        This routine should not be called directly but only
        through the bending_mode_correction routine.

        Parameters
        ----------
        modes_to_fit: list
            Bending modes to be fit.

        map: array
            Surface map as a 1D array in FEA coordinates.

        """

        # note that n_actuators is also n_bending modes in the U matrix.
        n_modes = self.force_space.data["U"].shape[1]
        n_nodes = self.force_space.data["U"].shape[0]

        # Note the defining equation is Af=z
        # where A is the influence fxns, f is the force, and z is the map.
        # We can't get the inverse of A, so we use SVD and then do a pseudo inverse

        V = self.force_space.data["V"]
        U = self.force_space.data["U"]
        sigma = self.force_space.data["W"]

        sigma_inv = np.diag(1 / sigma)
        # now filter out the bending modes that are not desired.
        removed = []
        for n in range(n_modes):
            if n not in modes_to_fit:
                sigma_inv[n] = 0.0
                removed.append(n)

        print(f"Not fitting modes: {removed}")

        # A = U * sigma * V-transpose
        # so solving for f's gives
        # f = V  sigma-inverse  U-transpose z = A_inv z

        A_inv = V @ sigma_inv @ U.T

        forces = A_inv @ map

        # Now do the force balancing
        f_balanced = self.force_space.data["af2lc"] @ forces

        A = U @ np.diag(sigma) @ V.T
        mode_fit = A @ f_balanced
        fit_map = mode_fit - np.mean(mode_fit)

        rmsForces_bal = np.sqrt(np.mean(f_balanced**2))

        # return rmsForces, fit, forces #  orig
        return rmsForces_bal, fit_map, forces

    def modalCorrection_orig(self, t, mapOnFEGrid):
        """
        Fits bending modes.
        Requires the data be sampled very specifically so as to match
        the fea nodes in the force_space matrix.
        This routine should not be called directly but only
        through the bending_mode_correction routine.
        """

        actuators = self.force_space.data["U"].shape[1]
        sinv = np.zeros((actuators, actuators))  # array for stiffness matrix
        maxBendingModes = t

        maskOnFEgrid = np.isfinite(mapOnFEGrid)

        for f in range(maxBendingModes):
            sinv[f, f] = 1.0 / self.force_space.data["S"][f, f]  # Create a stiffness matrix

            # Don't need to run the below code except for the last bending mode
            # as this is when the stiffness matrix is populated
            # Makes no difference in speed though.
            if f < (t - 1):
                continue
            # print(f'HERE for {f=}')

            bendCoef = np.dot(self.force_space.data["U"][:, :f].T, mapOnFEGrid[maskOnFEgrid])
            forceCoef = np.dot(
                sinv[:, :f], bendCoef
            )  # Multiply each bending mode coefficient by the stiffness to get how much of each force mode
            forces = np.dot(
                self.force_space.data["V"], forceCoef
            )  # Convert from force modes to actuator forces
            rmsForces = np.sqrt(
                np.dot(forces, forces) / actuators
            )  # Calculate RMS force of all actuator forces across the mirror
            # print(force_space['V'])

            forces_bal = np.dot(self.force_space.data["af2lc"], forces)
            forces_bal = -forces_bal
            rmsForces_bal = np.sqrt(np.dot(forces_bal, forces_bal) / actuators)

            zModeFit = np.dot(self.force_space.data["U"][:, :f], bendCoef)
            fit = zModeFit - np.mean(zModeFit)

        # return rmsForces, fit, forces #  orig
        return rmsForces_bal, fit, forces

    def bending_mode_correction(
        self, map, mask, n_modes, dx=None, coords=None, plots=False, rm_zerns=[0, 1, 2, 3, 6, 7], method=None
    ):
        """
        Removes bending modes from an UA produced borosilicate M1 mirror.
        The code utilizes a specific matrix provided with each mirror
        which is passed via the force_space parameter.

        The fitting requires the map to be sampled
        Because the Zernike terms of piston through coma are removed
        from displacing M2, these are also removed.

        Parameters
        ----------

        map: array
            Map in 2d space of surface error on a rectangular grid

        mask: array
            Boolean map in 2d space of surface error showing the pixels to be fit (1)
            and those to be ignored (0) on the same rectangular grid as the map parameter.

        n_modes: int
            Number of bending modes to be fit.

        Returns
        -------

        residual: array
            Map with Zernikes and bending modes removed.

        fitted_surf: array
            Bending mode map

        fitted_surf_mask: array
            Mask for fitted surface, which will use the entire mirror and not just the clear aperture.

        rmsForces: float

        forces: array
            Array of forces on each actuator

        """

        assert mask.dtype == bool, "Mask needs to be a dtype of boolean."

        # set terms in decomposed Zernike surface to zero to reconstruct a map without piston/tip/tilt/focus/coma (default)
        removed_map, zern_fit_map = remove_zerns(map, mask, rm_zerns)

        x_node = self.force_space.data["x_nodes"]  # x-location of nodes
        y_node = self.force_space.data["y_nodes"]  # y- location of nodes
        # fwhm = force_space["fwhm"]  # FIXME: Where does this come from?

        if coords != None:
            eff_idx = mask != 0
            x_vec = coords.x_grid[eff_idx]
            y_vec = coords.y_grid[eff_idx]
            z_vec = removed_map[eff_idx]

        else:
            raise OSError("Coords parameter has not been supplied")

        # bending mode fit and subtraction
        if method == None:
            # New method for interpolation and the new fitting method

            # works but creates errors at edges, can't ever get all values
            # fea_map = scipy.interpolate.interpn( (coords.x_grid[0,:], coords.y_grid[:,0]), removed_map, (y_node.flatten(), x_node.flatten()), bounds_error=False)

            from scipy.interpolate import RBFInterpolator

            y = (np.array((coords.x_grid[mask], coords.y_grid[mask]))).T
            new = np.array((x_node.flatten(), y_node.flatten())).T
            d = removed_map[mask].flatten()
            fea_map = RBFInterpolator(y, d, neighbors=5, kernel="linear")(new)

            # fea_map_on_grid = griddata(new, fea_map, (coords.x_grid, coords.y_grid))

            modes = list(range(0, n_modes))
            rmsForces, fit, forces = self.fit_bending(modes, fea_map)
        elif method == "orig":
            # resampling from WFE grid to point cloud space, which is the FEA nodes.
            fea_map = resampleGauss(x_vec, y_vec, z_vec, x_node, y_node, fwhm=0.2)
            rmsForces, fit, forces = self.modalCorrection_orig(n_modes, fea_map)

        else:
            raise OSError("Valid method is not provided.")

        # Fit remains in the fea map sampling, so must convert back to that of the original array
        # Surface is low order so no need to smoothing (meaning using the resampleGauss method)

        fitted_surf = griddata(
            (x_node.flatten(), y_node.flatten()),
            fit.flatten(),
            (coords.x_grid, coords.y_grid),
        )

        fitted_surf_mask = ~np.isnan(fitted_surf)

        residual = removed_map - fitted_surf

        if plots:
            import psd_utils

            psd_tools = psd_utils.PSDUtils()

            ncols = 5
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(20, ncols), ncols=ncols)
            # fig.suptitle('Original, Fitted, theoretical residuals, actual residuals')

            vals_mask = mask
            vals = map * vals_mask
            stats = psd_tools.get_map_stats(vals, vals_mask, report=False)
            ax1.imshow(vals)
            ax1.set_title("Original")
            ax1.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )

            vals_mask = mask
            vals = zern_fit_map * vals_mask
            stats = psd_tools.get_map_stats(vals, vals_mask, report=False)
            ax2.imshow(vals)
            ax2.set_title("Zernike Fit (PTTFC) to Map")
            ax2.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )

            vals_mask = mask
            vals = removed_map * vals_mask
            stats = psd_tools.get_map_stats(vals, vals_mask, report=False)
            ax3.set_title("Map - Zerns(PTTFC)")
            ax3.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax3.imshow(vals)

            vals_mask = mask
            vals = fitted_surf * vals_mask
            stats = psd_tools.get_map_stats(vals, vals_mask, report=False)
            ax4.set_title(f"Map of {n_modes} bending modes")
            ax4.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax4.imshow(vals)

            vals = residual
            vals_mask = mask
            stats = psd_tools.get_map_stats(vals, vals_mask, report=False)
            ax5.set_title(f"Map with PTTFC and \n {n_modes} bending modes removed")
            ax5.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f},\nForces(RMS)={rmsForces:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax5.imshow(vals)

            fname = "test_bending_mode_correction_plots.png"
            plt.savefig(TEST_OUTPUT_DATA_DIR.joinpath(fname))

            # plt.show()

        return residual, fitted_surf, fitted_surf_mask, rmsForces, forces

    def calc_moments(self, forces, assertion=True, warning_only=False, limit=1e-14):
        """Calculate the moments on the mirror for the given set of forces.

        Parameters
        ----------

        forces : array
            List of forces to be applied to the mirror

        assertion : bool
            Asserts that moments do not exceed the desired limit.

        limit : float
            The not to exceed limit of moments.

        warning_only : bool
            Only displays a warning instead of raising an error.

        """

        x_moment = np.sum(self.force_space.data["x_act"] @ forces)

        y_moment = np.sum(self.force_space.data["y_act"] @ forces)

        if assertion:
            errors = []
            try:
                val = np.abs(y_moment)
                assert val < limit, f"Y-moment of {val} on mirror exceeds {limit}."
            except AssertionError as e1:
                errors.append(e1)

            try:
                val = np.abs(x_moment)
                assert val < limit, f"X-moment of {val} on mirror exceeds {limit}."
            except AssertionError as e2:
                errors.append(e2)

            if warning_only:
                warnings.warn(f"Moments surpass limits: Reported errors are: {errors}", UserWarning)
            elif len(errors) > 0:
                raise AssertionError(f"Moments exceeded limit of {limit}. Reported errors are: {errors}")

        return x_moment, y_moment


def resampleGauss(x, y, z, xi, eta, fwhm=0.2):
    """Resamples z(x,y) onto zeta(xi,eta) using Gaussian kernel.

    zeta = resampleGauss(x,y,z,xi,eta,fwhm,r_max)

    Resamples z(x,y) onto zeta(xi,eta) using Gaussian kernel.

    x,y = vector or matrix of coordinates for original function
    z = vector or matrix of values of original function
    xi,eta = vector or matrix of coordinates for resampled function
    fwhm = FWHM of Gaussian kernel
    r_max = radius of kernel (include points (x,y) within r_max of (xi,eta))
    zeta = vector or matrix of values of resampled function

    """

    # TODO: Remove this function
    warnings.warn(f"resampleGauss using a FWHM of {fwhm}.")
    warnings.warn("resampleGauss will be deprecated soon, use interp2", DeprecationWarning)

    # x = x.flatten()
    # y = y.flatten()
    # z = z.flatten()
    # print(z.flatten().shape)

    denom = ((fwhm) ** 2) / (4 * np.log(2))  # denominator of the Gaussian argument
    r_maxSq = (0.3) ** 2  # defining the kernel size to sample points from

    # from original code - make vectors out of xi, eta
    # xi_is_matrix = isinstance(xi, np.ndarray)
    xi_is_matrix = np.ndim(xi) == 2

    if xi_is_matrix:
        m, n = xi.shape  # these values for m and n are gucci
        xiVec = xi.flatten()  # this is GOOD
        etaVec = eta.flatten()  # this is GOOD
        zetaVec = np.empty(m * n)
    else:
        m = len(xi)
        xiVec = xi
        etaVec = eta
        zetaVec = np.empty(m)

    mask = np.isfinite(z)
    # print(np.size(mask))
    x_vec = x[mask]
    y_vec = y[mask]
    z_vec = z[mask]  # something is going on here!!!
    zeta = np.empty(etaVec.shape)
    # print(y_vec)

    for k in range(len(etaVec)):
        dx = x_vec - xiVec[k]
        dy = y_vec - etaVec[k]
        drSq = dx**2 + dy**2
        kernelMask = drSq < r_maxSq
        ptsInKernel = np.sum(kernelMask)

        if ptsInKernel > 0:
            drSqInKernel = drSq[kernelMask]
            zInKernel = z_vec[kernelMask]
            gaussian = np.exp(-drSqInKernel / denom)
            sumWeight = np.sum(gaussian)
            sumZ = np.sum(zInKernel * gaussian)
            zetaVec[k] = sumZ / sumWeight

    if xi_is_matrix:
        zeta = zetaVec.reshape(m, n)
    else:
        zeta = zetaVec

    return zeta


def remove_zerns(map, mask, terms, plots=False, iterations=17):
    """Removes zernikes from a map

    Parameters
    ----------

    map: float
        2-d map of which the zernikes should be removed

    mask: int
        2-d map indicating which pixels should be fit (1), and which should be ignored (0).

    terms: array
        A 1-d array of numbers corresponding to the indices to be removed.
        E.g. 1,2,5,7

    plots: bool
        Show plots when executed?

    Returns
    ----------

    removed: array
        Map with desired zernikes removed

    zern_map: array
        Map with desired zernikes removed

    """
    # Fit map of zernikes
    # Need to fit up to the max number of terms, then extract them.

    # This is the correct call, but there is a bug in poppy that makes this fail due
    # to the npix parameter not being accepted by the arbitrary basis.
    # zerns = zernike.opd_expand_nonorthonormal(
    #     map, aperture=mask, nterms=np.max(terms), iterations=iterations,basis=zernike.arbitrary_basis)

    # There is a curious "bug" that if you fit the same number of terms you need then the fit is poor
    # I assume this is because with limited sampling and a discontinuous pupil the basis is not
    # orthonormal. Therefore, be sure to fit a few more terms
    full_fit_terms = np.max(terms) + (np.max(terms) / 3).astype(int)
    # always stay below 231
    full_fit_terms = 231 if full_fit_terms > 231 else full_fit_terms
    zerns = zernike.opd_expand_nonorthonormal(
        map, aperture=mask, nterms=full_fit_terms, iterations=iterations
    )

    # Create array with zernikes to keep and remove via multiplication
    keepers = np.array([1 if item in terms else 0 for item in range(full_fit_terms)])
    # print(f"{keepers=}")
    # print(f"{zerns=}")
    zerns_removed = zerns * keepers
    # print(f"zerns_removed are: {zerns_removed}")

    # Create a map that has the desired terms remaining and the selected terms removed.
    # Need to use the arbrary basis to use a defined pupil
    removal_map = zernike.compose_opd_from_basis(zerns_removed, aperture=mask, basis=zernike.arbitrary_basis)

    # create map that has the desired zernikes removed
    residual_map = map - removal_map

    if plots:
        import psd_utils

        psd_tools = psd_utils.PSDUtils()

        ncols = 3
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20, ncols), ncols=ncols)
        # fig.suptitle('Original, Fitted, theoretical residuals, actual residuals')
        vals = map * mask
        stats = psd_tools.get_map_stats(vals, mask, report=False)
        pos1 = ax1.imshow(vals)
        ax1.set_title("Original")
        ax1.annotate(
            f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
        )

        vals = (removal_map) * mask
        stats = psd_tools.get_map_stats(vals, mask, report=False)
        ax2.set_title("Fitted zerns")
        ax2.annotate(
            f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax2.imshow(vals)

        vals = residual_map * mask
        stats = psd_tools.get_map_stats(vals, mask, report=False)
        ax3.set_title("Residual Map")
        ax3.annotate(
            f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax3.imshow(vals)
        plt.show()

    return residual_map, removal_map
