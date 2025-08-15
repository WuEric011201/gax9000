"""
Merge all keysight_id_vgs.h5 per die, plot Id–Vgs (dir 2 & 3) to one PNG per die,
then tile those PNGs into a single wafer composite image (placed by x,y from filename).
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- HDF5 import/export helpers ----------
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

# ---------- Directory discovery ----------
def find_die_dirs(root):
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("die_x_"):
            yield p

def find_device_dirs(die_path):
    for name in sorted(os.listdir(die_path)):
        p = os.path.join(die_path, name)
        if os.path.isdir(p) and name.startswith("gax_"):
            yield p

# ---------- Merge per die ----------
def merge_die(die_path, program):
    dev_dirs = list(find_device_dirs(die_path))
    if not dev_dirs:
        raise RuntimeError(f"No device folders found in {die_path}")

    first = import_hdf5(os.path.join(dev_dirs[0], f"{program}.h5"))

    merged = {}
    array_keys = []
    for k, v in first.items():
        if isinstance(v, np.ndarray) and v.ndim > 1:
            array_keys.append(k)
            shape = (len(dev_dirs),) + v.shape  # (device, bias, sweep, points)
            merged[k] = np.full(shape, np.nan, dtype=v.dtype)
        else:
            merged[k] = v

    for i, d in enumerate(dev_dirs):
        data_i = import_hdf5(os.path.join(d, f"{program}.h5"))
        for k in array_keys:
            merged[k][i, ...] = data_i[k]

    out_dir = os.path.join(os.path.dirname(die_path), "_merged")
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, os.path.basename(die_path) + f"_{program}_merged.h5")
    # optional: export_hdf5(out_h5, merged)
    return merged, out_h5

# ---------- Plot per die (Id–Vgs) ----------
def plot_die_idvgs_one_png(die_name, merged, out_png_path,
                           vds_min=0.05, vds_max=0.8,
                           dir_indices=(2, 3),
                           use_abs=False,
                           ylog=False,
                           vds_tol=1e-3):
    """
    One PNG per die:
      X: V_GS, Y: I_D
      Only directions in dir_indices (default: 2 and 3)
      Include biases whose V_DS (at sweep start) falls in [vds_min, vds_max] (±vds_tol)
      Colors: vds≈vds_min -> navy; vds≈vds_max -> light-blue
      Styles: dir 2 = solid (Forward 2nd), dir 3 = dashed (Reverse 2nd)
    """
    vds = merged["v_ds"]   # shape: (dev, bias, sweep, pts)
    vgs = merged["v_gs"]
    id_ = merged["i_d"]
    id_plot = np.abs(id_) if use_abs else id_

    n_dev, n_bias, n_sweep, n_pts = id_plot.shape

    # style/labels for directions
    DIR_LABEL = {2: "Forward 2nd", 3: "Reverse 2nd"}
    DIR_STYLE = {2: "-", 3: "--"}

    # colors for the two target Vds
    VDS_COLOR = {
        "min": "navy",
        "max": "#065608",  # light-ish blue
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    # Build list of (bias_index, vds_value) within the window (read from first device for speed)
    read_dir = dir_indices[0]
    chosen = []
    for b in range(n_bias):
        vds_val = vds[0, b, read_dir, 0] if vds.ndim == 4 else vds[0, b]
        if np.isfinite(vds_val) and (vds_min - vds_tol) <= vds_val <= (vds_max + vds_tol):
            chosen.append((b, float(vds_val)))
    chosen.sort(key=lambda x: x[1])

    # keep track of how many devices actually produced a curve
    devices_with_any_curve = set()

    for (b, vds_val) in chosen:
        # choose color by proximity to ends of the window
        if abs(vds_val - vds_min) <= vds_tol:
            color = VDS_COLOR["min"]
        elif abs(vds_val - vds_max) <= vds_tol:
            color = VDS_COLOR["max"]
        else:
            color = None  # let matplotlib cycle

        for d in dir_indices:
            for dev in range(n_dev):
                x = vgs[dev, b, d, :]
                y = id_plot[dev, b, d, :]

                if not np.all(np.isfinite(x)) or not np.any(np.isfinite(y)):
                    continue

                # label just once per (bias, dir)
                label = None
                if dev == 0:
                    vtag = f"V_DS={vds_val:.3g} V"
                    dtag = f"{DIR_LABEL.get(d, f'dir {d}')}"
                    label = f"{vtag} • {dtag}"

                ax.plot(
                    x, y,
                    color=color,
                    linestyle=DIR_STYLE.get(d, "-"),
                    alpha=0.9 if color else 0.7,
                    linewidth=1.2,
                    label=label
                )
                devices_with_any_curve.add(dev)

    ax.set_xlabel("V_GS [V]")
    ax.set_ylabel("I_D [A]")
    # human-readable dirs in title
    dirs_str = ", ".join(DIR_LABEL.get(d, f"dir {d}") for d in dir_indices)
    ax.set_title(f"I_D–V_GS • {die_name} • {dirs_str}")

    if ylog:
        # only set log if data are positive; otherwise leave linear
        y_all = []
        for line in ax.lines:
            y_all.append(line.get_ydata())
        y_all = np.concatenate([np.asarray(y) for y in y_all]) if y_all else np.array([1.0])
        if np.any(y_all > 0):
            ax.set_yscale("log")

    ax.grid(True, which="both", ls=":")

    # Tight autoscale on Y (do not force limits)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, ncols=2, loc="best")
    ax.set_ylim(bottom=1e-12, top=1e-2)
    # ---- fixed top-right annotation ----
    n_dev_plotted = len(devices_with_any_curve)
    ax.text(
        0.98, 0.98,
        f"N devices = {n_dev_plotted}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )
    # ------------------------------------

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=200)
    plt.close(fig)

# ---------- Build wafer composite from per-die PNGs ----------
def build_wafer_composite_from_pngs(merged_root, out_path, pattern=None, bg_color=(255, 255, 255)):
    from PIL import Image  # pillow

    if pattern is None:
        pattern = re.compile(r"die_x_(-?\d+)_y_(-?\d+)_idvgs\.png")

    die_images = []
    for fname in os.listdir(merged_root):
        m = pattern.match(fname)
        if m:
            x, y = map(int, m.groups())
            die_images.append((x, y, os.path.join(merged_root, fname)))

    if not die_images:
        print(f"[wafer] No matching PNGs found in {merged_root}; skipping wafer composite.")
        return None

    xs = [x for x, _, _ in die_images]
    ys = [y for _, y, _ in die_images]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    first_img = Image.open(die_images[0][2])
    img_w, img_h = first_img.size
    first_img.close()

    grid_w = (max_x - min_x + 1) * img_w
    grid_h = (max_y - min_y + 1) * img_h

    canvas = Image.new("RGB", (grid_w, grid_h), bg_color)

    for x, y, path in die_images:
        img = Image.open(path)
        px = (x - min_x) * img_w
        py = (max_y - y) * img_h  # invert Y so larger Y is "up"
        canvas.paste(img, (px, py))
        img.close()

    canvas.save(out_path)
    print(f"[wafer] composite -> {out_path}")
    return out_path

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./scripts/wafer_map/W/multidie", help="Root folder containing die_x_* subfolders")
    ap.add_argument("--program", default="keysight_id_vgs", help="HDF5 program name (default: keysight_id_vgs)")
    ap.add_argument("--vds_min", type=float, default=0.05, help="Minimum V_DS to include (default 0.05 V)")
    ap.add_argument("--vds_max", type=float, default=0.8, help="Maximum V_DS to include (default 0.8 V)")
    ap.add_argument("--abs_current", default="true", help="Plot |Id| instead of Id (default off)")
    ap.add_argument("--log", default="true", help="Use log y-scale (default linear)")
    ap.add_argument("--vds_tol", type=float, default=1e-3, help="Tolerance when matching boundary biases")
    ap.add_argument("--skip_wafer_map", action="store_true", help="Only merge/plot per-die; skip composite wafer PNG")
    args = ap.parse_args()

    merged_roots_seen = set()

    for die_path in find_die_dirs(args.root):
        die_name = os.path.basename(die_path)

        merged, _ = merge_die(die_path, args.program)
        print(f"[{die_name}] merged")

        merged_root = os.path.join(os.path.dirname(die_path), "_merged")
        merged_roots_seen.add(merged_root)
        os.makedirs(merged_root, exist_ok=True)
        out_png = os.path.join(merged_root, f"{die_name}_idvgs.png")

        plot_die_idvgs_one_png(
            die_name,
            merged,
            out_png_path=out_png,
            vds_min=args.vds_min,
            vds_max=args.vds_max,
            dir_indices=(2, 3),         # Forward 2nd & Reverse 2nd
            use_abs=args.abs_current,
            ylog=args.log,
            vds_tol=args.vds_tol,
        )
        print(f"[{die_name}] plot -> {out_png}")

    if not args.skip_wafer_map:
        for merged_root in sorted(merged_roots_seen):
            wafer_out = os.path.join(merged_root, "wafer_map_combined.png")
            build_wafer_composite_from_pngs(merged_root, wafer_out)

if __name__ == "__main__":
    main()
