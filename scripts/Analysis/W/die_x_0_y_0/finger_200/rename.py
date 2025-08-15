import os

old_prefix = "gax_mod_fet_tlm_pmos_"
new_prefix = "gax_mod_fingered_fet_tlm_nmos_nf_4_"

for fname in os.listdir("."):
    if fname.startswith(old_prefix):
        new_name = new_prefix + fname[len(old_prefix):]
        print(f"Renaming: {fname} -> {new_name}")
        os.rename(fname, new_name)

print("Done.")
