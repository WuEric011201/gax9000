"""
Merge all keysight_id_vds.h5 under a single die folder and make basic plots.
Usage:
  python merge_and_plot_idvds.py "C:\path\to\scripts\data\die_x_0_y_0" --program keysight_id_vds
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# If you have controller.util.io, use it (it handles numpy dtype roundtrips nicely)
try:
    from controller.util.io import import_hdf5, export_hdf5
except ImportError:
    import h5py
    def import_hdf5(path):
        out = {}
        with h5py.File(path, "r") as f:
            def get_group(g):
                obj = {}
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        obj[k] = v[()]
                    else:
                        obj[k] = get_group(v)
                return obj
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    out[k] = v[()]
                else:
                    out[k] = get_group(v)
        return out
    def export_hdf5(path, data: dict):
        import h5py, numpy as np
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as f:
            def write_group(g, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        grp = g.create_group(k)
                        write_group(grp, v)
                    else:
                        v = np.array(v)
                        g.create_dataset(k, data=v)
            write_group(f, data)

def find_device_dirs(die_path):
    for name in os.listdir(die_path):
        p = os.path.join(die_path, name)
        if os.path.isdir(p) and name.startswith("gax_"):
            yield p

def merge_die(die_path, program):
    dev_dirs = sorted(find_device_dirs(die_path))
    if not dev_dirs:
        raise RuntimeError(f"No device folders found in {die_path}")

    # Load first to get shapes & keys
    first = import_hdf5(os.path.join(dev_dirs[0], f"{program}.h5"))

    # Build merged dict: stack arrays with ndim>1 along new axis 0
    merged = {}
    array_keys = []
    for k, v in first.items():
        if isinstance(v, np.ndarray) and v.ndim > 1:
            array_keys.append(k)
            shape = (len(dev_dirs),) + v.shape  # (device, bias, sweep, points)
            merged[k] = np.full(shape, np.nan, dtype=v.dtype)
        else:
            merged[k] = v  # copy scalars/1D metadata as-is from first

    # Fill arrays
    for i, d in enumerate(dev_dirs):
        data_i = import_hdf5(os.path.join(d, f"{program}.h5"))
        for k in array_keys:
            merged[k][i, ...] = data_i[k]

    # Save merged
    out_dir = os.path.join(os.path.dirname(die_path), "_merged")
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, os.path.basename(die_path) + f"_{program}_merged.h5")
    export_hdf5(out_h5, merged)
    return merged, out_h5

def plot_idvds(merged, out_dir, max_devices=12):
    """
    Plot I_D–V_DS families for each device at each V_GS bias.
    Expects merged keys: v_ds, v_gs, i_d with shape (device, bias, sweep, points).
    """
    vds = merged["v_ds"]
    vgs = merged["v_gs"]
    id_  = merged["i_d"]

    # Safety: take abs or leave as-is—your data may be negative for p-type
    # id_plot = np.abs(id_)
    id_plot = id_

    n_dev, n_bias, n_sweep, n_pts = id_plot.shape
    os.makedirs(out_dir, exist_ok=True)

    # 1) One PNG per device: all Vgs curves over Vds
    for dev in range(min(n_dev, max_devices)):
        plt.figure(figsize=(7,5))
        for b in range(n_bias):
            # Use first (forward) sweep if you have forward/reverse
            plt.plot(vds[dev, b, 0, :], id_plot[dev, b, 0, :], label=f"Vgs={vgs[dev, b, 0, 0]:.3g} V")
        plt.xlabel("V_DS [V]")
        plt.ylabel("I_D [A]")
        plt.title(f"I_D–V_DS (device {dev})")
        # plt.yscale("log")  # comment out if you prefer linear
        plt.grid(True, which="both", ls=":")
        plt.legend(fontsize=8, ncols=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"idvds_device_{dev}.png"), dpi=180)
        plt.close()

    # 2) Aggregate view: pick a single Vgs (e.g., last bias) and overlay all devices
    pick = n_bias - 1
    plt.figure(figsize=(7,5))
    for dev in range(min(n_dev, max_devices)):
        plt.plot(vds[dev, pick, 0, :], id_plot[dev, pick, 0, :], alpha=0.5)
    plt.xlabel("V_DS [V]")
    plt.ylabel("I_D [A]")
    plt.title(f"I_D–V_DS overlay @ V_GS≈{vgs[0, pick, 0, 0]:.3g} V (first {min(n_dev,max_devices)} devices)")
    # plt.yscale("log")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"idvds_overlay_vgs_index_{pick}.png"), dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("die_path", help="Path to a die folder containing gax_* subfolders")
    ap.add_argument("--program", default="keysight_id_vds", help="keysight_id_vds or keysight_id_vgs")
    ap.add_argument("--max_devices", type=int, default=12, help="Limit plotted devices per figure")
    args = ap.parse_args()

    merged, out_h5 = merge_die(args.die_path, args.program)
    print(f"Merged saved to: {out_h5}")

    out_plot_dir = os.path.join(os.path.dirname(args.die_path), "_merged", "plots_" + os.path.basename(args.die_path))
    plot_idvds(merged, out_plot_dir, max_devices=args.max_devices)
    print(f"Plots saved to: {out_plot_dir}")

if __name__ == "__main__":
    main()
