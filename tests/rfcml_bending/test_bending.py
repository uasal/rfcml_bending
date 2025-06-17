# tests the base configuration for each system is valid
import pathlib
from unittest import TestCase

from rfcml_bending import force_matrix
import numpy as np
import psd_utils
import scipy.io
from matplotlib import pyplot as plt
from poppy import zernike

from rfcml_bending.bending import Bending, remove_zerns

TEST_SUPPORT_DATA_DIR = pathlib.Path(__file__).parents[2].joinpath("tests", "data")
TEST_FORCESPACE_RW = TEST_SUPPORT_DATA_DIR.joinpath("stp_forcespace_RW_220928.mat")
TEST_FORCESPACE = TEST_SUPPORT_DATA_DIR.joinpath("stp_tel_166_forcespace_mkII.mat")


class TestBending(TestCase):
    """Tests for bending functions in bending.py."""

    def test_instantiate_bending(self):
        """Tests loading of file into namespace."""

        mat_file = TEST_FORCESPACE_RW

        Bending(mat_file)

    def test_remove_zerns(self):
        """Test that function removes desired zernikes."""
        psd_tools = psd_utils.PSDUtils()

        dims = (256 * 2, 256 * 2)
        dx = 10
        coords = psd_tools.coord_arrays(dims, dx=dx)

        mask = np.zeros(dims)
        # Leave a ~1 pixel border all around
        mask[coords.r_grid < ((dims[0] - 2) / 2 * dx)] = 1
        mask[coords.r_grid < (30 * dx)] = 0
        mask[np.abs(coords.x_grid) < (4 * dx)] = 0  # spider
        mask[np.abs(coords.y_grid) < (4 * dx)] = 0  # spider

        rndm_num_gen = np.random.default_rng(seed=12)
        nzerns = 25
        input_zern_coeff = rndm_num_gen.uniform(low=-1.0, high=1.0, size=nzerns) * 100

        print(f"{input_zern_coeff=}")

        # Need to use an arbitrary basis to support a discontinuous aperture
        map = zernike.compose_opd_from_basis(input_zern_coeff, aperture=mask, basis=zernike.arbitrary_basis)
        # map = zernike.opd_from_zernikes(input_zern_coeff, aperture=mask,npix=max(dims))

        _expect = np.sum(mask)
        _actual = np.sum(~np.isnan(map))
        self.assertTrue(
            _expect == _actual,
            msg=f"Total of mask is {_expect:0.1f}, but map has {_actual:0.1f}. They should be equal",
        )

        terms_to_remove = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 15]
        # terms_to_remove = [0,1,2,3,4,5,6,7,8,9,10,11]

        resultant_map, removal_map = remove_zerns(map, mask, terms_to_remove, plots=False)

        # Map should be what is left, which is the terms that were not removed
        zerns_resid = [0 if item in terms_to_remove else 1 for item in range(nzerns)]
        zerns_remaining = zerns_resid * input_zern_coeff
        print(f"zerns_remaining: {zerns_remaining}")
        # Again, use the arbitrary basis
        theory_resid = zernike.compose_opd_from_basis(
            zerns_remaining,
            aperture=mask,
            basis=zernike.arbitrary_basis,
        )

        diagnosis_plots = False
        if diagnosis_plots:
            ncols = 5
            (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(20, ncols), ncols=ncols)
            # fig.suptitle('Original, Fitted, theoretical residuals, actual residuals')
            vals = map
            stats = psd_tools.get_map_stats(vals, mask, report=False)
            ax1.imshow(vals, cmap="Blues")
            ax1.set_title("Original")
            ax1.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )

            vals = map - resultant_map
            stats = psd_tools.get_map_stats(vals, mask, report=False)
            ax2.set_title(f"Fitted zerns:\n {terms_to_remove}")
            ax2.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax2.imshow(vals, cmap="Blues")

            vals = theory_resid
            stats = psd_tools.get_map_stats(vals, mask, report=False)
            ax3.set_title("theoretical resids")
            ax3.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax3.imshow(vals, cmap="Blues")

            vals = resultant_map
            stats = psd_tools.get_map_stats(vals, mask, report=False)
            ax4.set_title("Actual resids")
            ax4.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax4.imshow(vals, cmap="Blues")

            # Note that this may be mostly noise, so PtoV and RMS are
            # probably not ideal
            vals = theory_resid - resultant_map
            stats = psd_tools.get_map_stats(vals, mask, report=False)
            ax5.set_title("Theory-Actual resids")
            ax5.annotate(
                f"PtoV={stats.ptov:0.1f}\nRMS={stats.sigma:0.1f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
            )
            ax5.imshow(vals, cmap="Blues")

        # Check that theoretical residuals and actual residuals are within ~2% of each other
        # in PtoV and RMS
        stats_resultant = psd_tools.get_map_stats(resultant_map, mask, report=False)
        stats_theory_resid = psd_tools.get_map_stats(theory_resid, mask, report=False)

        _got = np.abs(stats_resultant.ptov)
        _expect = np.abs(stats_theory_resid.ptov)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = 2  # Somewhat arbitrary as it depends on the number of terms

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected PtoV of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        _got = np.abs(stats_resultant.rms)
        _expect = np.abs(stats_theory_resid.rms)
        _perc_diff = 100 * (_expect - _got) / _expect
        _criteria = 2  # Somewhat arbitrary as it depends on the number of terms

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

    def test_bending_mode_correction(self):
        """Test that function removes desired zernikes.
        Note that this is currently for an unobscured pupil only.
        Using a non-continuous surface requires changing the fitting type."""
        psd_tools = psd_utils.PSDUtils()

        # Create an array which is 6.6m, and we'll make the OD=6.42m, ID=1.38m
        # values are from stp_reference_data as of 2024-10-14
        od = 6.42
        id = 1.38
        dims = (256 * 4, 256 * 4)
        dx = 6.6 / dims[1]
        coords = psd_tools.coord_arrays(dims, dx=dx)

        mask = np.zeros((dims), dtype=bool)
        mask[coords.r_grid < od / 2] = 1
        mask[coords.r_grid < id / 2] = 0

        rndm_num_gen = np.random.default_rng(seed=11)
        # fit 200 zerns
        nzerns = 200

        # use a ~1/f fall off... but it's in zerns so not quite the same.
        input_zern_coeff = (
            1000 * rndm_num_gen.uniform(low=-1.0, high=1.0, size=nzerns) * 1 / (np.arange(nzerns) + 1)
        )
        map = zernike.compose_opd_from_basis(input_zern_coeff, aperture=mask, npix=max(dims))

        modes = 33

        mat_file = TEST_FORCESPACE
        # force_space = load_force_space(mat_file)
        bending = Bending(mat_file)

        residual, fitted_surf, fitted_surf_mask, rms_forces, forces = bending.bending_mode_correction(
            map, mask, modes, plots=True, coords=coords
        )

        _expect = np.sum(mask)
        _actual = np.sum(~np.isnan(residual))
        self.assertTrue(
            _expect == _actual,
            msg=f"Input map has {_expect:0.1f} valid pixels,"
            "but map has {_actual:0.1f}. They should be equal",
        )

    def test_bending_solvay_orig(self):
        """Test that the code reproduces what Solvay had originally produced.
        This uses Becca's original fitting code and forcespace, as well
        as a map supplied by Solvay."""
        psd_tools = psd_utils.PSDUtils()

        mat_file = TEST_SUPPORT_DATA_DIR.joinpath("syn_wfe_09061000.mat")  # surface error
        input_map = scipy.io.loadmat(mat_file)["wavefront"]  # wavefront error

        # Solvay reports surface error in matlab, and wavefront error in python
        # so dividing wfe by 2 before assertions
        input_map /= 2.0

        map = input_map
        mask = ~np.isnan(map)
        map[~mask] = 0

        stats_map = psd_tools.get_map_stats(map, mask, report=False)

        input_rms = 32  # [nm] surface rms for input map - from Solvay (python) 2024-10-14

        _got = np.abs(stats_map.rms)
        _expect = np.abs(input_rms)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / 32) * 100  # should be identical, except there is no decimal, so must be within 0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS of {_expect:0.2f}, but got {_got:0.2f},"
            "a % diff of {_perc_diff:0.2f},"
            "where success is {_criteria:0.2f}",
        )

        # Create an array which is 6.6m, and we'll make the OD=6.42m, ID=1.38m
        # values are from stp_reference_data as of 2024-10-14
        od = 6.42
        id = 1.38
        dx = 6.5 / input_map.shape[0]
        coords = psd_tools.coord_arrays(map.shape, dx=dx)

        # reduce the mask to match the clear aperture
        mask[coords.r_grid > od / 2] = 0
        mask[coords.r_grid < id / 2] = 0

        # First check the zernikes removal
        # per Solvay - Z1-Z4,Z7-8 map has of 19nm RMS (python)
        terms_to_remove = [0, 1, 2, 3, 6, 7]
        resultant_map, removed_map = remove_zerns(map, mask, terms_to_remove, plots=False)
        stats_zerns_removed = psd_tools.get_map_stats(resultant_map, mask, report=False)

        _got = np.abs(stats_zerns_removed.rms)
        _expect = 19
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # FIXME:Not enough significant figures, so will be within ~0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on map with Zernikes removed of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        # Now remove the bending modes
        modes = 33

        mat_file = TEST_FORCESPACE_RW
        bending = Bending(mat_file)

        # matlab
        residual, fitted_surf, fitted_surf_mask, rms_forces, forces = bending.bending_mode_correction(
            map,
            mask,
            modes,
            plots=True,
            coords=coords,
            method="orig",
        )

        # Check that theoretical residuals and actual residuals are within ~10% of each other
        # they will be a little different as the masks are different.

        # 33 bending mode fitted map from Solvay is 18nm RMSx
        stats_resultant_bending = 18
        # Final map
        stats_resultant_theory = 5

        stats_bending = psd_tools.get_map_stats(fitted_surf, mask, report=False)
        stats_resultant = psd_tools.get_map_stats(residual, mask, report=False)

        _got = np.abs(stats_resultant.rms)
        _expect = np.abs(stats_resultant_theory)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # Not enough significant figures, so will be within ~0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on residual map of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        _got = np.abs(stats_bending.rms)
        _expect = np.abs(stats_resultant_bending)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # Again, lack significant figures

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on bending map of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        # Check forces, RMS and Max
        # from Solvay: 4.3 N in python and the max force which is 10.7 N
        # new interpolation gives a lower force value but we're unsure
        # which is correct, so setting to <5N and < 11N.
        # new interpolation

        _got = rms_forces
        _expect = 5
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        # _criteria = (0.1 / _expect) * 100  # Use sig-figs, but might be too tight

        self.assertTrue(
            _got < _expect,
            msg=f"Expected RMS of forces of less than {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f}.",
        )

        _got = np.max(forces)
        _expect = 11
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        # _criteria = (0.1 / _expect) * 100  # Use sig-figs, but might be too tight

        self.assertTrue(
            _got <= _expect,
            msg=f"Expected max force of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f}.",
        )

    def test_bending_solvay(self):
        """Test that the code reproduces what Solvay has produced.
        This uses the same input map, but a modified fitting code.
        This also uses a more standard forcespace based on those
        created by Steve West."""
        psd_tools = psd_utils.PSDUtils()

        mat_file = TEST_SUPPORT_DATA_DIR.joinpath("syn_wfe_09061000.mat")  # surface error
        input_map = scipy.io.loadmat(mat_file)["wavefront"]  # wavefront error

        # Solvay reports surface error in matlab, and wavefront error in python
        # so dividing wfe by 2 before assertions
        input_map /= 2.0

        map = input_map
        mask = ~np.isnan(map)
        map[~mask] = 0

        stats_map = psd_tools.get_map_stats(map, mask, report=False)

        input_rms = 32  # [nm] surface rms for input map - from Solvay (python) 2024-10-14

        _got = np.abs(stats_map.rms)
        _expect = np.abs(input_rms)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / 32) * 100  # should be identical, except there is no decimal, so must be within 0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS of {_expect:0.2f}, but got {_got:0.2f},"
            "a % diff of {_perc_diff:0.2f},"
            "where success is {_criteria:0.2f}",
        )

        # Create an array which is 6.6m, and we'll make the OD=6.42m, ID=1.38m
        # values are from stp_reference_data as of 2024-10-14
        od = 6.42e3
        id = 1.38e3
        dx = 6.5e3 / input_map.shape[0]
        coords = psd_tools.coord_arrays(map.shape, dx=dx)

        # reduce the mask to match the clear aperture
        mask[coords.r_grid > od / 2] = 0
        mask[coords.r_grid < id / 2] = 0

        # First check the zernikes removal
        # per Solvay - Z1-Z4,Z7-8 map has of 19nm RMS (python)
        terms_to_remove = [0, 1, 2, 3, 6, 7]
        resultant_map, removed_map = remove_zerns(map, mask, terms_to_remove, plots=False)
        stats_zerns_removed = psd_tools.get_map_stats(resultant_map, mask, report=False)

        _got = np.abs(stats_zerns_removed.rms)
        _expect = 19
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # FIXME:Not enough significant figures, so will be within ~0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on map with Zernikes removed of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        # Now remove the bending modes
        modes = 33

        # Load Steve's version of the force matrix
        mat_file = TEST_FORCESPACE

        # force_space = load_force_space(mat_file)
        bending = Bending(mat_file)

        # matlab
        residual, fitted_surf, fitted_surf_mask, rms_forces, forces = bending.bending_mode_correction(
            map, mask, modes, plots=True, coords=coords
        )

        # Check that theoretical residuals and actual residuals are within ~10% of each other
        # they will be a little different as the masks are different.

        # 33 bending mode fitted map from Solvay is 18nm RMSx
        stats_resultant_bending = 18
        # Final map
        stats_resultant_theory = 5

        stats_bending = psd_tools.get_map_stats(fitted_surf, mask, report=False)
        stats_resultant = psd_tools.get_map_stats(residual, mask, report=False)

        _got = np.abs(stats_resultant.rms)
        _expect = np.abs(stats_resultant_theory)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # Not enough significant figures, so will be within ~0.5

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on residual map of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        _got = np.abs(stats_bending.rms)
        _expect = np.abs(stats_resultant_bending)
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        _criteria = (0.5 / _expect) * 100  # Again, lack significant figures

        self.assertTrue(
            _perc_diff <= _criteria,
            msg=f"Expected RMS on bending map of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f},"
            "where success is {_criteria:0.1f}",
        )

        # Check forces, RMS and Max
        # from Solvay: 4.3 N in python and the max force which is 10.7 N
        # new interpolation gives a lower force value but we're unsure
        # which is correct, so setting to <5N and <11N

        _got = rms_forces
        _expect = 5
        # _perc_diff = 100 * np.abs(_expect - _got) / _expect
        # _criteria = (0.1 / _expect) * 100  # Use sig-figs, but might be too tight

        self.assertTrue(
            _got <= _expect,
            msg=f"Expected RMS of forces of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f}.",
        )

        _got = np.max(forces)
        _expect = 11
        _perc_diff = 100 * np.abs(_expect - _got) / _expect
        # _criteria = (0.1 / _expect) * 100  # Use sig-figs, but might be too tight

        self.assertTrue(
            _got <= _expect,
            msg=f"Expected max force of {_expect:0.1f},"
            "but got {_got:0.1f}, a % diff of {_perc_diff:0.1f}.",
        )

    def test_calc_moments(self):
        """Applies moments and verifies loads are balanced."""

        # Load Steve's version of the force matrix
        mat_file = TEST_FORCESPACE
        bending = Bending(mat_file)
        # force_space = load_force_space(mat_file)

        # let's arbitrarily apply a force at 2 actuators
        force1 = 2
        force2 = 3
        i1 = 25
        i2 = 100
        bending.force_space.data["FEA_act_ids"][i1]
        bending.force_space.data["FEA_act_ids"][i2]

        # calculate expectations
        moment_x_exp = (
            force1 * bending.force_space.data["x_act"][i1] + force2 * bending.force_space.data["x_act"][i2]
        )
        moment_y_exp = (
            force1 * bending.force_space.data["y_act"][i1] + force2 * bending.force_space.data["y_act"][i2]
        )

        # Create a forces vector:
        forces = np.zeros(len(bending.force_space.data["FEA_act_ids"]))
        forces[i1] = force1
        forces[i2] = force2

        # Check that only a warning is thrown

        with self.assertWarns(UserWarning):
            mom_x, mom_y = bending.calc_moments(forces, assertion=True, warning_only=True)
            self.assertAlmostEqual(moment_x_exp, mom_x)
            self.assertAlmostEqual(moment_y_exp, mom_y)

        # Check that an error is raised
        with self.assertRaises(AssertionError):
            mom_x, mom_y = bending.calc_moments(forces, assertion=True)

        # Now load balance and check no errors are thrown.
        # Note that the acceptable limit is unknown
        # Assuming 0.1% of input.
        f_balanced = bending.force_space.data["af2lc"] @ forces

        limit = np.abs(0.001 * np.min([moment_x_exp, moment_y_exp]))
        mom_x, mom_y = bending.calc_moments(f_balanced, limit=limit)


# tmp=TestBending()
# tmp.test_instantiate_bending()
# tmp.test_bending_mode_correction()
# tmp.test_bending_solvay_orig()
# tmp.test_bending_solvay()
# tmp.test_bending_solvay()
# tmp.test_calc_moments()
