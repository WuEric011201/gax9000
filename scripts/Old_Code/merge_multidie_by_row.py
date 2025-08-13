"""
Merge multidie modules measurement sweep data.
Folder structure:
    die_x_-1_y_0
        gax_r0_c0_2023_07_01_09_54_50
        gax_r0_c1_2023_07_01_09_54_51
        gax_r0_c2_2023_07_01_09_54_52
        gax_r1_c0_2023_07_01_09_54_53
        gax_r1_c1_2023_07_01_09_54_55
        gax_r1_c2_2023_07_01_09_54_56
        ...
    die_x_0_y_0
        gax_r0_c0_2023_07_01_09_54_50
        gax_r0_c1_2023_07_01_09_54_51
        gax_r0_c2_2023_07_01_09_54_52
        gax_r1_c0_2023_07_01_09_54_53
        gax_r1_c1_2023_07_01_09_54_55
        gax_r1_c2_2023_07_01_09_54_56
        ...
    die_x_1_y_0
        gax_r0_c0_2023_07_01_09_54_50
        gax_r0_c1_2023_07_01_09_54_51
        gax_r0_c2_2023_07_01_09_54_52
        gax_r1_c0_2023_07_01_09_54_53
        gax_r1_c1_2023_07_01_09_54_55
        gax_r1_c2_2023_07_01_09_54_56
        ...

We want to merge this data into a `_merged` folder:
    _merged
        die_x_-1_y_0_r0.h5
        die_x_-1_y_0_r1.h5
        die_x_0_y_0_r0.h5
        die_x_0_y_0_r1.h5
        die_x_1_y_0_r0.h5
        die_x_1_y_0_r1.h5

"""

from collections import defaultdict
from dataclasses import dataclass

@dataclass
class ModuleMeasurement:
    """Individual module measurement in a die.
    Filename format of each measurement is
        gax_[module]_[index]_[timestamp]
        
        - module: variable length
        - index: number
        - timestamp: 2023_07_01_04_39_24 (always 19 length)

        e.g.
        gax_mod_fet_tlm_nmos_lc_0.12_lch_0.10_lov_0.06_gateasym_0.00_w_4.0_6_2023_07_01_04_39_24
    """
    filename: str
    timestamp: str
    row: int
    col: int

    def from_filename(filename: str):
        timestamp = filename[-19:]
        tag = filename[4:-20]
        
        # find row and col
        chunks = tag.split("_")
        row = None
        col = None
        for s in chunks:
            if s.startswith("r"):
                row = int(s[1:])
            elif s.startswith("c"):
                col = int(s[1:])

        return ModuleMeasurement(
            filename=filename,
            timestamp=timestamp,
            row=row,
            col=col,
        )

def merge_multidie_module_data(
    path: str,
    program: str = "keysight_id_vds",
):
    import numpy as np
    import os
    from controller.util.io import import_hdf5, export_hdf5, export_mat
    
    path_out = os.path.join(path, "_merged")
    os.makedirs(path_out, exist_ok=True)

    print(os.listdir(path))
    for d in os.listdir(path):
        if not d.startswith("die"):
            continue
        path_die = os.path.join(path, d)
        print(d, path_die)

        # map die measurements row => col list
        die_measurements = defaultdict(list)

        for measurement_path in os.listdir(path_die):
            if not measurement_path.startswith("gax"):
                continue
            measurement = ModuleMeasurement.from_filename(measurement_path)
            die_measurements[measurement.row].append(measurement)

        # sort measurements by row and col
        for row, measurements in die_measurements.items():
            measurements.sort(key=lambda x: x.col)
        
        # merge measurements into single .h5 file
        # TODO: merge all measurements, not just idvg
        for row, measurements in die_measurements.items():
            num_devices = len(measurements)

            # load first measurement to find data shape
            data0 = import_hdf5(os.path.join(path_die, measurements[0].filename, f"{program}.h5"))
            data_shape = data0["i_d"].shape

            # make merged data for all keys (i_d, v_gs, v_ds, etc.)
            data_merged = {}
            for k, v in data0.items():
                if type(v) is np.ndarray:
                    if len(v.shape) > 1:
                        # alloc new array for all files, will be filled later
                        data_merged[k] = np.full((num_devices, data_shape[0], data_shape[1], data_shape[2]), np.nan)
                    else: # common properties
                        data_merged[k] = v
                else:
                    data_merged[k] = v
            
            # insert numpy array data into merged arrays
            for i, measured in enumerate(measurements):
                data_i = import_hdf5(os.path.join(path_die, measured.filename, f"{program}.h5"))
                for k, v in data_i.items():
                    if type(v) is np.ndarray and len(v.shape) > 1:
                        data_merged[k][i,:] = v
            
            path_out_merged = os.path.join(path_out, f"{d}_r{row}")
            export_hdf5(path_out_merged + ".h5", data_merged)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Merge multidie modules sweep .h5 files.")

    parser.add_argument(
        "path",
        metavar="path",
        type=str,
        help="Folder with multidie data files (die_x_0_y_0, die_x_1_y_0, etc.)"
    )

    parser.add_argument(
        "--program",
        metavar="program",
        type=str,
        default="keysight_id_vds",
        help="Measurement program type name, e.g. keysight_id_vgs or keysight_id_vds"
    )

    args = parser.parse_args()

    path = args.path
    program = args.program

    print(f"PATH: f{path}")
    print(f"PROGRAM TYPE: f{program}")

    merge_multidie_module_data(
        path=path,
        program=program,
    )
