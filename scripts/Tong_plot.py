import h5py
import numpy as np
import matplotlib.pyplot as plt

# === Path to your HDF5 output file ===
h5_path = "scripts/data/keysight_id_vds.h5"  # change if needed

# === Load data from HDF5 ===
with h5py.File(h5_path, 'r') as f:
    v_ds = np.array(f['v_ds'])      # shape: (num_bias, num_sweeps, num_points)
    v_gs = np.array(f['v_gs'])      # same shape
    i_d = np.array(f['i_d'])        # drain current, same shape

# === Choose only the first sweep (index 0), for simplicity ===
# Shape: (num_bias, num_points)
v_gs_vals = v_gs[:, 0, 0]
v_ds_vals = v_ds[0, 0, :]  # same for each bias point

# === Plot Id vs Vds for each constant Vgs ===
plt.figure(figsize=(8, 6))

for i, vgs in enumerate(v_gs_vals):
    plt.plot(v_ds[i, 0, :], i_d[i, 0, :], label=f"Vgs = {vgs:.2f} V")

plt.xlabel("Vds [V]")
plt.ylabel("Id [A]")
plt.title("Id-Vds Characteristics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
