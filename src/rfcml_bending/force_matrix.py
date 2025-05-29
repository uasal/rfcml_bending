import logging
from pathlib import Path

import numpy as np
import scipy.io
from scipy.interpolate import griddata

logger = logging.getLogger("__force_matrix__")


class ForceSpace:
    """Creates class to handle force spaces.
    Each forcespace matrix is a little different,
    so this class standardizes the format.
    """

    def __init__(self, filename) -> None:
        """Populate and return the object based on the input name."""

        path_to_file = Path(filename)

        # verify existence
        if not path_to_file.exists():
            raise ValueError(f"File {filename} does not exist")

        # declare standardized dictionary of parameters
        self.data = {
            "U": None,
            "V": None,
            "sigma": None,
            "af2lc": None,
            "x_nodes": None,
            "y_nodes": None,
            "FEA_act_ids": None,
            "x_act": None,
            "y_act": None,
        }

        self.filename = None

        # Note that forcespace_512 was an early copy of the file
        # from Steve West by Becca, but changes had been made to the columns
        # to support her fitter (which is the modalCorrection method in this
        # package).
        if path_to_file.name == "stp_forcespace_RW_220928.mat" or path_to_file.name == "forcespace_512.mat":
            logging.info("Loading STP force space matrix")
            self.load_stp_rw(path_to_file)

        elif path_to_file.name == "um_42_forcespace.mat":
            logging.info("Loading Ultramarine force space matrix")
            self.load_stp(path_to_file)  # FIXME

        elif path_to_file.name == "six_five_100_forcespace_mkIII.mat":
            logging.info("Loading theoretical 100 actuator telescope force space matrix")
            # same as ultramarine format
            self.load_stp(path_to_file)

        elif path_to_file.name == "stp_tel_166_forcespace_mkII.mat":
            logging.info("Loading Steve West's STP force space")
            self.load_stp(path_to_file)

        else:
            raise ValueError(f"File {filename} is not supported.")

    def load_steve_west_base(self, raw):
        """Function loads the parameters which are common among the force spaces
        produced by Steve West."""

        self.data["W"] = raw["s_Un"][:, 0]  # Scaling factor to be applied to U
        self.data["U"] = raw["U_m"]
        self.data["V"] = raw["V_m"]
        self.data["af2lc"] = raw["mf2lc"]  # 166x166
        self.data["lc2mf"] = raw["lc2mf"]  # 166x166
        self.data["A"] = raw["f2Un"]  # 17058x166 (rowsXcolumns)
        self.data["x_nodes"] = raw["node_xy"][:, 0]  # chance that this is backwards.
        self.data["y_nodes"] = raw["node_xy"][:, 1]
        self.data["x_act"] = raw["FEA_act_xy"][:, 0]
        self.data["y_act"] = raw["FEA_act_xy"][:, 1]
        # New formalism
        self.data["sigma"] = raw["s_Un"][:, 0]  # Scaling factor to be applied to U
        self.data["U"] = raw["U_m"]
        self.data["V"] = raw["V_m"]

        # Change FEA_act_ids into a simple array and not array of arrays
        _final = np.zeros(len(raw["FEA_act_ids"]), dtype=int)
        for i, v in enumerate(raw["FEA_act_ids"]):
            _final[i] = v[0]

        self.data["FEA_act_ids"] = _final

    def load_um(self, file_path):
        """Populates the force_space.data dictionary from the UM specific force space matrix.
        FIXME -- this is nearly the same as the stp one minus one key.
        Parameters
        ----------

        file_path: obj
            Pathlib path object containing the path to the force space matrix.


        """
        raw = self.load(file_path)

        self.load_steve_west_base(raw)

        self.filename = file_path.name

        self.data["W"] = raw["s_Un"][:, 0]  # Scaling factor to be applied to U
        self.data["U"] = raw["U_m"]
        self.data["V"] = raw["V_m"]
        self.data["af2lc"] = raw["mf2lc"]  # 166x166
        self.data["lc2mf"] = raw["lc2mf"]  # 166x166
        self.data["A"] = raw["f2Un"]  # 17058x166 (rowsXcolumns)
        self.data["x_nodes"] = raw["node_xy"][:, 0]  # chance that this is backwards.
        self.data["y_nodes"] = raw["node_xy"][:, 1]
        self.data["x_act"] = raw["FEA_act_xy"][:, 0]
        self.data["y_act"] = raw["FEA_act_xy"][:, 1]
        # New formalism
        self.data["sigma"] = raw["s_Un"][:, 0]  # Scaling factor to be applied to U
        self.data["U"] = raw["U_m"]
        self.data["V"] = raw["V_m"]

        # Change FEA_act_ids into a simple array and not array of arrays
        _final = np.zeros(len(raw["FEA_act_ids"]), dtype=int)
        for i, v in enumerate(raw["FEA_act_ids"]):
            _final[i] = v[0]

        self.data["FEA_act_ids"] = _final

        self.data["S"] = raw["f2Un"]  # FIXME!!
        self.data["U"] = raw["U_m"]
        self.data["V"] = raw["V_m"]
        self.data["af2lc"] = raw["mf2lc"]

        self.data["x_nodes"] = raw["node_xy"][:, 0]  # chance that this is backwards.
        self.data["y_nodes"] = raw["node_xy"][:, 1]

        # Change FEA_act_ids into a simple array and not array of arrays
        _final = np.zeros(len(raw["FEA_act_ids"]), dtype=int)
        for i, v in enumerate(raw["FEA_act_ids"]):
            _final[i] = v[0]

        self.data["FEA_act_ids"] = _final

        self.filename = file_path.name

    def load_stp(self, file_path):
        """Populates the force_space.data dictionary from the STP specific force space matrix.

        Parameters
        ----------

        file_path: obj
            Pathlib path object containing the path to the force space matrix.

        """
        raw = self.load(file_path)

        self.load_steve_west_base(raw)

        self.filename = file_path.name

    def load_stp_rw(self, file_path):
        """Populates the force_space.data dictionary from the STP specific forceSpace matrix.

        Parameters
        ----------

        file_path: obj
            Pathlib path object containing the path to the forceSpace matrix.

        """
        _raw = self.load(file_path)

        self.data["S"] = _raw["S"]
        self.data["U"] = _raw["U"]
        self.data["V"] = _raw["V"]
        self.data["af2lc"] = _raw["RW_af2lc"]
        self.data["x_nodes"] = _raw["xnode"][:, 0]
        self.data["y_nodes"] = _raw["ynode"][:, 0]

        # Change FEA_act_ids into a simple array and not array of arrays
        _final = np.zeros(len(_raw["FEA_act_ids"]), dtype=int)
        for i, v in enumerate(_raw["FEA_act_ids"]):
            _final[i] = v[0]

        self.data["FEA_act_ids"] = _final

        self.filename = file_path.name

    # Load force_space vector
    def load(self, path_to_file):
        """Loading the mirror lab force_space *.mat file to be read into Python.
        Mirror map is 512x512 pixels.

        Each forcespace matrix is a little different,
        """

        force_space = scipy.io.loadmat(path_to_file)

        return force_space

    def fea_to_grid(self, input_map, coords=None, x=None, y=None, dim=None, method="linear"):
        """Resamples FEA-based map to a desired grid.
        If no x and y inputs are provided then a similarly
        sampled rectilinear grid is output.

        input_map: array
            1d array of an input map that has the same mapping as the fea data.

        coords: array
            Coordinate array of the desired output.

        x : array
            Optional input array for x-values. Not implemented.

        y : array
            Optional input array for y-values. Not implemented.
        """

        # Derive grids if none are provided
        if coords is None:
            raise ValueError("Coordinate array is required.")
        else:
            x_vec = np.ndarray.flatten(coords.x_grid)
            y_vec = np.ndarray.flatten(coords.y_grid)

        if len(input_map) != len(self.data["x_nodes"]):
            raise ValueError(
                f"""Input map (len={len(input_map)})
                and coordinate arrays (length={len(x_vec)})
                are not the same size."""
            )

        if method == "gauss":
            # # resampling from WFE grid to point cloud space, which is the FEA nodes.
            # map = bending.resampleGauss(force_space.data['x_nodes'], force_space.data['y_nodes'], input_map,
            #                             x_vec,
            #                             y_vec,
            #                             fwhm=0.2)
            raise NotImplementedError("gaussian interpolation not implemented")

        if method == "linear":
            map = griddata(
                (self.data["x_nodes"], self.data["y_nodes"]), input_map, (coords.x_grid, coords.y_grid)
            )

        if coords:
            return map
        else:
            return map, x_vec, y_vec
