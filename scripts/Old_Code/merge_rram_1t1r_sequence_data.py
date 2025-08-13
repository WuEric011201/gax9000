def merge_rram_1t1r_sequence_data(
    path_in,
    path_out,
):
    """Merge all 1T1R repeated sequence data into single file.
    Each data in folder should be .h5 file in format:
        keysight_rram_1t1r_sequence_0.h5
        keysight_rram_1t1r_sequence_1.h5
        keysight_rram_1t1r_sequence_2.h5
        ...
    Take all these files, sort by number at end, then merge into
    single .h5 file, write to path_out.
    """
    import os
    import json
    import numpy as np
    from controller.util.io import import_hdf5, export_hdf5, export_mat

    # get list of sorted files by sequence run number
    data_files = [ p for p in os.listdir(path_in) if p.endswith(".h5") ]
    data_filenames = [ p[:-3] for p in data_files ]
    data_ids = np.array([ int(p.split("_")[-1]) for p in data_filenames ])
    idx_sorted = np.argsort(data_ids)
    data_files_sorted = [ data_files[i] for i in idx_sorted ]

    # merge sorted files
    
    # load first file, figure out data shape and insert non-zero shapes
    num_steps = len(data_files_sorted)
    data0 = import_hdf5(os.path.join(path_in, data_files_sorted[0]))
    data_shape = data0["i_d"].shape
    print(f"data_shape = {data_shape}")
    data_merged = {}
    for k, v in data0.items():
        if type(v) is np.ndarray:
            if len(v.shape) > 1:
                # alloc new array for all files, fill first row
                data_merged[k] = np.full((num_steps, data_shape[0], data_shape[1], data_shape[2]), np.nan)
                data_merged[k][0,:] = v
            else: # common properties
                data_merged[k] = v
        else:
            data_merged[k] = v
    
    for i, p in enumerate(data_files_sorted[1:]):
        print(i, p)
        data_step = import_hdf5(os.path.join(path_in, p))
        for k, v in data_step.items():
            if type(v) is np.ndarray and len(v.shape) > 1:
                data_merged[k][i+1,:] = v
    
    # save output
    n_start = 0
    n_end = num_steps
    path_out_merged = os.path.join(path_out, f"keysight_rram_1t1r_sequence_merged_{n_start}_to_{n_end}")
    export_hdf5(path_out_merged + ".h5", data_merged)
    export_mat(path_out_merged + ".mat", data_merged)
    

if __name__ == "__main__":
    import argparse
    import os
    import json
    from controller.util.io import export_hdf5, export_mat
    from controller.backend import ControllerSettings

    parser = argparse.ArgumentParser(description="Merge RRAM 1T1R sequence measurement files into single .h5 file.")

    parser.add_argument(
        "path_in",
        metavar="path_in",
        type=str,
        help="Folder with individual data files"
    )
    parser.add_argument(
        "path_out",
        metavar="path_out",
        type=str,
        help="Folder to put merged data file"
    )

    args = parser.parse_args()

    path_in = args.path_in
    path_out = args.path_out

    print(f"PATH INPUTS: f{path_in}")
    print(f"PATH OUTPUT: f{path_out}")

    merge_rram_1t1r_sequence_data(
        path_in=path_in,
        path_out=path_out,
    )
