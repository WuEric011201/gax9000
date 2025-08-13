"""
Merge multidie modules measurement sweep data.
Folder structure:
data:
    die_x_-1_y_0
        gax_[module1_0]_2023_07_01_09_54_50
        gax_[module1_1]_2023_07_01_09_54_51
        gax_[module2_0]_2023_07_01_09_54_52
        gax_[module2_1]_2023_07_01_09_54_53
        ...
    die_x_0_y_0
        gax_[module1_0]_2023_07_01_09_54_60
        gax_[module1_1]_2023_07_01_09_54_61
        gax_[module2_0]_2023_07_01_09_54_62
        gax_[module2_1]_2023_07_01_09_54_63
        ...
    die_x_1_y_0
        gax_[module1_0]_2023_07_01_09_54_70
        gax_[module1_1]_2023_07_01_09_54_71
        gax_[module2_0]_2023_07_01_09_54_72
        gax_[module2_1]_2023_07_01_09_54_73
        ...

We want to merge this data into a `_merged` folder:
    _merged
        die_x_-1_y_0_[module1].h5
        die_x_-1_y_0_[module2].h5
        die_x_0_y_0_[module1].h5
        die_x_0_y_0_[module2].h5
        die_x_1_y_0_[module1].h5
        die_x_1_y_0_[module2].h5

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
    module: str
    index: int
    timestamp: str

    def from_filename(filename: str):
        timestamp = filename[-19:]
        module_index = filename[4:-20]
        
        # find last _ to split module_index into module and index
        split = module_index.rindex("_")
        module = module_index[:split]
        index = int(module_index[split+1:])

        return ModuleMeasurement(
            filename=filename,
            module=module,
            index=index,
            timestamp=timestamp,
        )

def merge_multidie_module_data(
    path: str,
    program: str = "keysight_id_vgs",
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

        # gather all module measurements done in die by module name
        die_measurements = defaultdict(list)

        for measurement_path in os.listdir(path_die):
            if not measurement_path.startswith("gax"):
                continue
            measurement = ModuleMeasurement.from_filename(measurement_path)
            die_measurements[measurement.module].append(measurement)

        # sort measurements by index
        for mod, measurements in die_measurements.items():
            measurements.sort(key=lambda x: x.index)
        
        # merge measurements into single .h5 file
        # TODO: merge all measurements, not just idvg
        for mod, measurements in die_measurements.items():
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
            
            path_out_merged = os.path.join(path_out, f"{d}_{mod}")
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
