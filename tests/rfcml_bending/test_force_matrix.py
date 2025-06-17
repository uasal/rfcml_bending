from pathlib import Path

import numpy as np
import psd_utils
import pytest

from rfcml_bending import force_matrix

# TODO: Update to use files in UASAL_archive

TEST_FILE_UM = Path("tests/data/um_42_forcespace.mat")
# For reference only, but will soon not be supported
# TEST_FILE_STP = Path('tests/data/stp_forcespace_RW_220928.mat')
TEST_FILE_STP = Path("tests/data/stp_tel_166_forcespace_mkII.mat")
# For reference only, but is supported.
TEST_FILE_STP_100 = Path("tests/data/six_five_100_forcespace_mkIII.mat")


EXPECTED_TYPES = {
    "U": None,
    "V": None,
    "af2lc": None,
    "x_nodes": None,
    "y_nodes": None,
    "FEA_act_ids": None,
}

# S is only used in the RW force_space which is being deprecated
# EXPECTED_TYPES["S"]={"type":np.ndarray, "ndims":2}
EXPECTED_TYPES["U"] = {"type": np.ndarray, "ndims": 2}
EXPECTED_TYPES["V"] = {"type": np.ndarray, "ndims": 2}
EXPECTED_TYPES["af2lc"] = {"type": np.ndarray, "ndims": 2}
EXPECTED_TYPES["x_nodes"] = {"type": np.ndarray, "ndims": 1}
EXPECTED_TYPES["y_nodes"] = {"type": np.ndarray, "ndims": 1}
EXPECTED_TYPES["FEA_act_ids"] = {"type": np.ndarray, "ndims": 1}


def check_types(force_space):
    """Checks the types of the data for each key in the force_space output."""
    # TODO: this should catch each assertion in a list then report all issues

    assert force_space.filename is not None, "No filename specified in force space"

    # exceptions=[]

    for key in EXPECTED_TYPES:
        # check each key in the excepted types
        # TODO: probably a better way to do this maybe with
        # pytest.mark.parametrize
        got = force_space.data[key]
        expected = EXPECTED_TYPES[key]["type"]
        assert isinstance(got, expected), (
            f"Expected type {expected} but got {got} for data attribute {key} in {force_space.filename}"
        )
        got = force_space.data[key].ndim
        expected = EXPECTED_TYPES[key]["ndims"]
        assert got == expected, (
            f"Expected {expected} but got {got} for data attribute in {force_space.filename}"
        )

    return True


@pytest.mark.skipif(not TEST_FILE_UM.exists, reason="No Ultramarine force space (.mat) file found.")
def test_um():
    """Test loading of UM force space"""

    force_space = force_matrix.ForceSpace(TEST_FILE_UM)

    check_types(force_space)


@pytest.mark.skipif(not TEST_FILE_STP.exists, reason="No STP force space (.mat) file found.")
def test_stp():
    """Test loading of UM force space"""

    force_space = force_matrix.ForceSpace(TEST_FILE_STP)

    check_types(force_space)


@pytest.mark.skipif(not TEST_FILE_STP.exists, reason="No Ultramarine force space (.mat) file found.")
def test_fea_to_grid():
    """tests function to go from fea space to grid space in 1 line.
    Test uses plot produced by Steve West's documentation."""

    force_space = force_matrix.ForceSpace(TEST_FILE_STP)

    # Just use a influence fxn mode as an example map
    # A has rows of surface, columns of influence (actuator)
    inf_fxn = 0
    input_map = force_space.data["A"][:, inf_fxn]

    psd_tools = psd_utils.PSDUtils()

    xrange = np.max(force_space.data["x_nodes"]) - np.min(force_space.data["x_nodes"])  # mm

    n_nodes = len(force_space.data["x_nodes"])

    # area of a square the encompasses a circle of radius r is 2r*2r=4r^2
    # so a square will have 4r^2 / pi r^2 more samples = 1.273
    # so a side has to be ~sqrt(1.273) longer

    side = int(np.sqrt(n_nodes) * np.sqrt(1.273))
    dims = (side * 1, side * 1)
    dx = xrange / dims[0]
    coords = psd_tools.coord_arrays(dims, dx=dx, offset_x=0.0, offset_y=0.0)

    output_map = force_space.fea_to_grid(input_map, coords=coords)

    output_2d = np.reshape(output_map, dims)

    plots = True
    if plots:
        # for diagnostics only
        from matplotlib import pyplot as plt

        plt.imshow(output_2d, origin="lower")
        plt.colorbar()
        plt.title(
            f"Influence function {inf_fxn}, which is actuator {force_space.data['FEA_act_ids'][inf_fxn]}"
        )


# test_fea_to_grid()
