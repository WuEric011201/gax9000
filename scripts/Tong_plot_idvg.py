import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the .h5 file
with h5py.File('scripts/gax_r1_c1_2025_08_06_02_05_55/keysight_id_vgs.h5', 'r') as f:
    # Explore top-level keys
    print("Top-level keys:", list(f.keys()))

    # Example: access datasets (update these names to match your actual file)
    i_d = np.array(f['i_d'])       # shape: (nbias, ndirections, npoints)
    i_g = np.array(f['i_g'])
    v_ds = np.array(f['v_ds'])
    v_gs = np.array(f['v_gs'])

# Print shape info
num_bias, num_directions, num_points = i_d.shape
print(f"num_bias = {num_bias}, num_directions = {num_directions}, num_points = {num_points}")

# Create plots
fig_id, ax_id = plt.subplots()
ax_id.set_yscale('log')
ax_id.set_xlabel('Vgs [V]')
ax_id.set_ylabel('Id [A]')
ax_id.set_title('Id vs Vgs')

fig_ig, ax_ig = plt.subplots()
ax_ig.set_yscale('log')
ax_ig.set_xlabel('Vgs [V]')
ax_ig.set_ylabel('Ig [A]')
ax_ig.set_title('Ig vs Vgs')

# Loop through biases and directions
for b in range(num_bias):
    for d in range(num_directions):
        vds_val = v_ds[b, d, 0] if v_ds.ndim == 3 else v_ds[b, d]  # depends on shape
        vgs = v_gs[b, d, :] if v_gs.ndim == 3 else v_gs[b, d]
        id_ = np.abs(i_d[b, d, :])
        ig_ = np.abs(i_g[b, d, :])

        ax_id.plot(vgs, id_, label=f"Vds={vds_val:.2f}V dir={d}")
        ax_ig.plot(vgs, ig_, label=f"Vds={vds_val:.2f}V dir={d}")

# Optional legends
ax_id.legend()
ax_ig.legend()

# Show plots
plt.show()
