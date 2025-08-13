import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- optional: use controller.util.io if available ----------
try:
    from controller.util.io import import_hdf5
except ImportError:
    import h5py
    def import_hdf5(path):
        out = {}
        with h5py.File(path, "r") as f:
            def pull(g):
                obj = {}
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        obj[k] = v[()]
                    else:
                        obj[k] = pull(v)
                return obj
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    out[k] = v[()]
                else:
                    out[k] = pull(v)
        return out

# ---------------- parsing ----------------
F = r"(-?\d+(?:\.\d+)?)"

# Pattern A: "normal" naming (includes gateasym)
RE_NORMAL = re.compile(
    rf"^gax_.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_gateasym_{F}_w_{F}_(\d+)_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)

# Pattern B: "wsweep" naming (often omits gateasym, e.g. ..._lov_0.06_w_0.0_3_YYYY_MM_DD_hh_mm_ss)
RE_WSWEEP = re.compile(
    rf"^gax_.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_w_{F}_(\d+)_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)

def parse_device_dirname(dirname):
    """Return dict with type, lc, lch, lov, gateasym, w, device_id or None if not matched."""
    m = RE_NORMAL.match(dirname)
    if m:
        dev_type, lc, lch, lov, gateasym, w, dev_id = m.groups()
        return dict(
            type=dev_type.lower(), lc=float(lc), lch=float(lch), lov=float(lov),
            gateasym=float(gateasym), w=float(w), device_id=int(dev_id)
        )
    m = RE_WSWEEP.match(dirname)
    if m:
        dev_type, lc, lch, lov, w, dev_id = m.groups()
        # gateasym not present in name; default to 0.0 (adjust if your data says otherwise)
        return dict(
            type=dev_type.lower(), lc=float(lc), lch=float(lch), lov=float(lov),
            gateasym=0.0, w=float(w), device_id=int(dev_id)
        )
    return None

def scan_device_dirs(root):
    recs = []
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            meta = parse_device_dirname(name)
            if meta:
                meta["dirname"] = name
                meta["path"] = os.path.join(dirpath, name)
                recs.append(meta)
    return recs

# ---------------- selection helpers ----------------
def filter_by_constants(records, dev_type, lc, lch, lov, gateasym=None, tol=1e-9):
    out = []
    for r in records:
        if dev_type and r["type"] != dev_type:
            continue
        if abs(r["lc"] - lc) > tol: continue
        if abs(r["lch"] - lch) > tol: continue
        if abs(r["lov"] - lov) > tol: continue
        if gateasym is not None and abs(r["gateasym"] - gateasym) > tol: continue
        out.append(r)
    return out

def choose_one_per_value(records, key):
    """Pick the first record per unique value of `key` (e.g., per w)."""
    seen = {}
    for r in sorted(records, key=lambda x: x["dirname"]):
        val = r[key]
        if val not in seen:
            seen[val] = r
    return [(k, seen[k]) for k in sorted(seen)]

# ---------------- data extraction ----------------
def find_bias_index_for_vds(vds, dir_index, target_vds):
    nbias = vds.shape[0]
    vals = np.array([vds[b, dir_index, 0] for b in range(nbias)])
    return int(np.nanargmin(np.abs(vals - target_vds)))

def load_idvgs_curve(h5_path, dir_indices, target_vds, use_abs=False, average_dirs=False):
    if not os.path.exists(h5_path):
        return None, None
    try:
        d = import_hdf5(h5_path)
        i_d = np.array(d["i_d"])
        v_gs = np.array(d["v_gs"])
        v_ds = np.array(d["v_ds"])
    except Exception:
        return None, None

    if i_d.ndim != 3:
        return None, None

    got = []
    for d_idx in dir_indices:
        if d_idx < 0 or d_idx >= i_d.shape[1]:
            continue
        b = find_bias_index_for_vds(v_ds, d_idx, target_vds)
        vgs = v_gs[b, d_idx, :]
        idc = i_d[b, d_idx, :]
        if use_abs:
            idc = np.abs(idc)
        got.append((vgs, idc))

    if not got:
        return None, None
    if average_dirs and len(got) > 1:
        ref = got[0][0]
        if all(np.array_equal(ref, g[0]) for g in got[1:]):
            ids = np.stack([g[1] for g in got], axis=0)
            return ref, np.nanmean(ids, axis=0)
    return got[0]

# ---------------- main plotting function (W sweep) ----------------
def sweep_plot_w(root, dev_type, lc, lch, lov, gateasym=None,
                 program="keysight_id_vgs", target_vds=0.8, dir_indices=(2,),
                 average_dirs=False, use_abs=False, logy=False,
                 out_dir="./", fname_prefix=None):
    """
    Sweep all available 'w' values (one device per w), plot Id–Vgs on ONE axes.
    - dev_type: "nmos" | "pmos" (required)
    - lc, lch, lov: required constants
    - gateasym: optional (defaults to 0.0 for wsweep names; pass explicitly if you need strict filtering)
    """
    recs = scan_device_dirs(root)
    subset = filter_by_constants(recs, dev_type, lc, lch, lov, gateasym)
    if not subset:
        raise SystemExit("No devices matched type/constants.")

    # choose one device per w
    choices = choose_one_per_value(subset, "w")
    if not choices:
        raise SystemExit("No W values found under given filters.")

    # plot
    plt.figure(figsize=(9, 6))
    good = 0
    for w_val, rec in choices:
        h5 = os.path.join(rec["path"], f"{program}.h5")
        vgs, idc = load_idvgs_curve(
            h5, dir_indices=dir_indices, target_vds=target_vds,
            use_abs=use_abs, average_dirs=average_dirs
        )
        if vgs is None:
            print(f"[skip] could not load {h5}")
            continue
        plt.plot(vgs, idc, label=f"w={w_val:g}")
        good += 1

    if good == 0:
        raise SystemExit("Failed to load any curves.")

    plt.xlabel("V_GS [V]")
    plt.ylabel("|I_D| [A]" if use_abs else "I_D [A]")
    ttl_const = f"type={dev_type}, lc={lc:g}, lch={lch:g}, lov={lov:g}" + (f", gateasym={gateasym:g}" if gateasym is not None else "")
    plt.title(f"Id–Vgs @ Vds≈{target_vds:g} V, dir(s) {','.join(map(str,dir_indices))} • sweep=w\n{ttl_const}")
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both", ls=":")
    ax = plt.gca(); ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()

    # Build descriptive filename
    prefix = fname_prefix or f"{dev_type}_w_sweep"
    parts = [prefix, f"lc_{lc:g}", f"lch_{lch:g}", f"lov_{lov:g}"]
    if gateasym is not None:
        parts.append(f"gateasym_{gateasym:g}")
    parts.append(f"vds_{target_vds:g}")
    parts.append(f"dirs_{'-'.join(map(str,dir_indices))}")
    fname = "_".join(parts) + ".png"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] wrote -> {out_path}")
    return out_path

def sweep_plot_w_same_plot(root, dev_type, lc, lch, lov, gateasym=None,
                 program="keysight_id_vgs", target_vds=0.8, dir_indices=(2,),
                 average_dirs=False, use_abs=False, logy=False,
                 out_dir="./", fname_prefix=None):
    """
    Sweep all available 'w' values, plot Id–Vgs on ONE axes.
    For each w, plot ALL devices that match (e.g., 8 devices) using the SAME color.
    Different dir_indices are distinguished by line style but keep the same color for that w.
    """
    recs = scan_device_dirs(root)
    subset = filter_by_constants(recs, dev_type, lc, lch, lov, gateasym)
    if not subset:
        raise SystemExit("No devices matched type/constants.")

    # group all matching devices by w (do not collapse to one device)
    from collections import defaultdict
    by_w = defaultdict(list)
    for r in subset:
        by_w[r["w"]].append(r)

    if not by_w:
        raise SystemExit("No W values found under given filters.")

    # assign color that darkens as w increases
    w_values = sorted(by_w.keys())
    cmap = plt.cm.viridis  # choose any sequential colormap: viridis, cividis, Blues, etc.
    
    # normalize w into [0, 1] for the colormap
    w_min, w_max = min(w_values), max(w_values)
    if w_max > w_min:
        w_to_color = {w: cmap((w - w_min) / (w_max - w_min)) for w in w_values}
    else:
        # all w are same, just use one color
        w_to_color = {w: cmap(0.5) for w in w_values}


    # line styles per direction index
    dir_styles = {0: "-", 1: "--", 2: "-", 3: ":"}

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for w_val in w_values:
        color = w_to_color[w_val]
        # only label first curve we draw for this w to keep legend one-entry-per-w
        labeled_this_w = False

        # all devices for this w (e.g., 8 devices: device_id=0..7)
        for rec in sorted(by_w[w_val], key=lambda r: r.get("device_id", 0)):
            h5 = os.path.join(rec["path"], f"{program}.h5")

            for d_idx in dir_indices:
                vgs, idc = load_idvgs_curve(
                    h5, dir_indices=[d_idx],
                    target_vds=target_vds,
                    use_abs=use_abs,
                    average_dirs=False  # keep directions separate
                )
                if vgs is None:
                    continue

                plt.plot(
                    vgs, idc,
                    color=color,
                    linestyle=dir_styles.get(d_idx, "-"),
                    alpha=0.8, linewidth=1.0,
                    label=(f"w={w_val:g}" if not labeled_this_w else None)
                )
                labeled_this_w = True
                plotted_any = True

    if not plotted_any:
        raise SystemExit("Found devices but failed to load any curves (check program/Vds/dirs).")

    plt.xlabel("V_GS [V]")
    plt.ylabel("|I_D| [A]" if use_abs else "I_D [A]")
    parts = [f"type={dev_type}", f"lc={lc:g}", f"lch={lch:g}", f"lov={lov:g}"]
    if gateasym is not None: parts.append(f"gateasym={gateasym:g}")
    plt.title("Id–Vgs • sweep=w @ Vds≈{:.3g} V • dirs {}".format(
        target_vds, ",".join(map(str, dir_indices))
    ) + "\n" + ", ".join(parts))
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both", ls=":")
    ax = plt.gca(); ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
    plt.legend(ncols=3, fontsize=8)
    plt.tight_layout()

    # descriptive filename
    prefix = fname_prefix or f"{dev_type}_w_sweep"
    fname_bits = [prefix, f"lc_{lc:g}", f"lch_{lch:g}", f"lov_{lov:g}"]
    if gateasym is not None:
        fname_bits.append(f"gateasym_{gateasym:g}")
    fname_bits.append(f"vds_{target_vds:g}")
    fname_bits.append(f"dirs_{'-'.join(map(str, dir_indices))}")
    out_path = os.path.join(out_dir, "_".join(fname_bits) + ".png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] wrote -> {out_path}")
    return out_path



# ---------------- CLI wrapper ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep and plot Id–Vgs across all W values (one device per W).")
    ap.add_argument("--root", default="./scripts/die1-1_full_sweep/w_sweep", help="Root folder to scan (recursively)")
    ap.add_argument("--type", default= "nmos", choices=["nmos","pmos"], help="Device type parsed from folder name")
    ap.add_argument("--lc", default = "0.2", type=float)
    ap.add_argument("--lch", default = "0.12", type=float)
    ap.add_argument("--lov", default = "0.06", type=float)
    ap.add_argument("--gateasym", type=float, default=None, help="Optional; if omitted, don't enforce")
    ap.add_argument("--program", default="keysight_id_vgs")
    ap.add_argument("--target_vds", type=float, default=0.8)
    ap.add_argument("--dir_indices", default="2", help="Comma list, e.g. '2' or '2,3'")
    ap.add_argument("--avg_dirs", action="store_true")
    ap.add_argument("--abs_current", default="true")
    ap.add_argument("--logy", default="true")
    ap.add_argument("--out", default="./scripts/die1-1_full_sweep/_config_summary", help="Output PNG path")
    ap.add_argument("--fname_prefix", default=None)
    args = ap.parse_args()

    dir_idx = tuple(int(s) for s in args.dir_indices.split(",") if s.strip() != "")
    sweep_plot_w_same_plot(
        root=args.root,
        dev_type=args.type,
        lc=args.lc, lch=args.lch, lov=args.lov, gateasym=args.gateasym,
        program=args.program, target_vds=args.target_vds, dir_indices=dir_idx,
        average_dirs=args.avg_dirs, use_abs=args.abs_current, logy=args.logy,
        out_dir=args.out, fname_prefix=args.fname_prefix
    )
