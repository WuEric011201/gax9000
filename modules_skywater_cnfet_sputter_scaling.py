"""
Generates modules .toml file for skywater cnfet sputter scaling mask.
Used for gax9000 auto probe modules sweep. 
"""

import argparse
import os
import csv
import tomli
from dataclasses import dataclass
from jinja2 import Environment, BaseLoader

parser = argparse.ArgumentParser(description="Generate toml modules config for gax9000 autoprobing.")

parser.add_argument(
    "-m",
    "--modules",
    dest="modules",
    action="store_true",
    help="Generates modules.toml file"
)

parser.add_argument(
    "-t",
    "--tests",
    dest="tests",
    action="store_true",
    help="Generates test sweep .toml files"
)

args = parser.parse_args()

print(args)

PATH_MODULES_LIST = os.path.join("build", "skywater_cnfet_sputter_scaling_v1_modules.csv")

PATH_OUT = os.path.join("build", "measurements")
os.makedirs(PATH_OUT, exist_ok=True)

PATH_GAX9000_FET_MODULES_TOML = os.path.join(PATH_OUT, "skywater_cnfet_sputter_scaling_v1_fet_modules.toml")

# modules toml jinja template as string
MODULES_TOML_TEMPLATE = (
"""
# skywater / mit cnfet scaling mask
# designed by acyu@mit.edu, july 2022
# for gax9000 auto probe modules sweep

[modules]
{%- for module in modules %}
"{{ module.name }}" = { name = "{{ module.name }}", x = {{ module.x }}, y = {{ module.y }} }
{%- endfor %}

"""
)

@dataclass
class LayoutModule:
    """Layout module for autoprobing."""
    type: str
    name: str
    x: float
    y: float
    count: int

# main module names
modules = []

# all single devices, format is: "module_{INDEX}"
# e.g. there are 4 FETs / module, so for a module we would add in
# module_0, module_1, module_2, module_3
all_devices = []

# parse modules from modules file created with gds
with open(PATH_MODULES_LIST, "r") as f:
    reader = csv.reader(f)
    headers = next(reader, None)
    for row in reader:
        module_type = row[1]
        module_name = row[2]
        module_x = float(row[3])
        module_y = float(row[4])
        module_count = int(row[5])

        # Parse FET and FINGERED FET modules
        if module_type == "FET" or module_type == "FINGERED FET":
            # add in main module name
            modules.append(LayoutModule(
                type=module_type,
                name=module_name,
                x=module_x,
                y=module_y,
                count=module_count,
            ))

            num_per_module = 4 # number of FETs vertically in each module column
            dy = 270 # distance between FETs vertically
            print(f"module_type: {module_type}, module_name: {module_name}, module_x: {module_x}, module_y: {module_y}, module_count: {module_count}")
            
            # NOTE: numbers arranged from top (0) to bottom (N)
            # e.g. for 6 modules, module 0 is top, module (6*4)-1 = 23 is bottom
            # so we count in reverse 23...0 while incrementing y + dy
            y = module_y
            for r in reversed(range(0, module_count * num_per_module)):
                all_devices.append(LayoutModule(
                    type=module_type,
                    name=f"{module_name}_{r}",
                    x=module_x,
                    y=y,
                    count=module_count,
                ))
                y += dy

        # Parse cmos modules (TODO)
        # if module_type == "CMOS":
        #     print(f"module_type: {module_type}, module_name: {module_name}, module_x: {module_x}, module_y: {module_y}, module_count: {module_count}")

# export modules.toml file
if args.modules:
    # load jinja toml modules template and output modules list
    toml_template = Environment(loader=BaseLoader).from_string(MODULES_TOML_TEMPLATE)
    toml_modules = toml_template.render(modules=all_devices)
    with open(PATH_GAX9000_FET_MODULES_TOML, "w+") as f:
        f.write(toml_modules)

# =============================================================================
# export test sweeps toml config files 
# =============================================================================
TEST_SWEEP_TOML_TEMPLATE = (
"""
# generates modules file .toml config
modules_file = "skywater_cnfet_sputter_scaling_v1_fet_modules.toml"

[sweep]
modules = [
    {%- for module in modules %}
    "{{ module.name }}",
    {%- endfor %}
]
"""
)

TEST_SWEEP_MULTIDIE_TOML_TEMPLATE = (
"""
### sweep die coordinates clockwise
dies = [
    # center
    [0, 0],
    # first ring
    [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1],
    # 2nd ring corners (if needed)
    # [0, 2], [2, 0], [0, -2], [-2, 0],
    # second ring
    [0, 2], [1, 2], [2, 2], [2, 1],
    [2, 0], [2, -1], [2, -2], [1, -2],
    [3, 2], [3, 1], [3, 0], [3, -1], [3, -2],
    [0, -2], [-1, -2], [-2, -2], [-2, -1],
    [-2, 0], [-2, 1], [-2, 2], [-1, 2],
    [-3, 2], [-3, 1], [-3, 0], [-3, -1], [-3, -2],
    # third ring edges (outer edges)
    [-4, -1], [-4, 0], [-4, 1],
    [-1, 3], [0, 3], [1, 3],
    [4, 1], [4, 0], [4, -1],
    [1, -3], [0, -3], [-1, -3],
]

### uncomment to enable per-die height compensation
### adjusts probe height based on interpolated measured wafer heightmap
height_compensation_file = "height_offset.toml"

### reference to all modules in .toml
modules_file = "skywater_cnfet_sputter_scaling_v1_fet_modules.toml"

### reference to sweep config
sweep_file = "{{ sweep_name }}"
"""
)

sweep_template_single = Environment(loader=BaseLoader).from_string(TEST_SWEEP_TOML_TEMPLATE)
sweep_template_multidie = Environment(loader=BaseLoader).from_string(TEST_SWEEP_MULTIDIE_TOML_TEMPLATE)

def generate_test_sweep_config(
    name: str,                        # name of config => output path to save sweep config
    mod_prefixes: list[str],          # list of module names prefixes to output in sweep config
    contains: list[list[str]] = None, # list of lists of string chunks contained in module name
    how_many_to_measure = 3,          # number of FETs to actually measure total in column
    num_per_module = 4,               # number of FETs vertically in each module column
    dy = 270,                         # distance between FETs vertically
    multidie = False,                 # whether to use multidie sweep config
) -> str:
    """Helper to generate test sweep toml config.
    """
    modules_quick_fet_check = []
    for mod in modules:
        # Parse FET and FINGERED FET modules
        if not any([mod.name.startswith(s) for s in mod_prefixes]):
            continue
        # check if module names contains chunks in string groups
        # e.g. contains = list of lists of lch or lc to
        # filter by specific module parameters
        if contains is not None:
            has_chunk = True
            for contains_group in contains:
                if not any([s in mod.name for s in contains_group]):
                    has_chunk = False
                    break
            if not has_chunk:
                continue
        
        # NOTE: numbers arranged from top (0) to bottom (N)
        # e.g. for 6 modules, module 0 is top, module (6*4)-1 = 23 is bottom
        # so we count in reverse 23...0 while incrementing y + dy
        i = 0
        y = mod.y
        for r in reversed(range(0, mod.count * num_per_module)):
            modules_quick_fet_check.append(LayoutModule(
                type=mod.type,
                name=f"{mod.name}_{r}",
                x=mod.x,
                y=y,
                count=mod.count,
            ))
            y += dy

            i += 1
            if i >= how_many_to_measure:
                break
    
    # generate sweep config
    sweep_config_toml = sweep_template_single.render(modules=modules_quick_fet_check)
    path_toml = os.path.join(PATH_OUT, name + ".toml")
    with open(path_toml, "w+") as f:
        f.write(sweep_config_toml)
    
    if multidie: # generate additional multi die test and sweep configs
        multidie_test_config_toml = sweep_template_multidie.render(sweep_name=name + ".toml")
        path_multidie_test = os.path.join(PATH_OUT, "multidie_" + name + ".toml")
        with open(path_multidie_test, "w+") as f:
            f.write(multidie_test_config_toml)
    
    return name

# create list of all test files

tests = [
    # DEBUG TESTS: check module sweep method works
    generate_test_sweep_config(
        name="debug_test_modules",
        mod_prefixes=["mod_fet_tlm_nmos"],
        how_many_to_measure=2,
    ),

    generate_test_sweep_config(
        name="debug_multidie_test_modules",
        mod_prefixes=[
            "mod_fet_tlm_nmos_lc_0.20_lch_0.16",
            "mod_fet_tlm_nmos_lc_0.16_lch_0.12",
        ],
        how_many_to_measure=2,
        multidie=True,
    ),

    generate_test_sweep_config(
        name="debug_multidie_quick_lc_lch_check",
        mod_prefixes=["mod_fet_tlm_nmos"],
        # contains=[
        #     ["lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10", "lch_0.08", "lch_0.07", "lch_0.06"],
        #     ["lc_0.16", "lc_0.12", "lc_0.08"],
        # ],
        contains=[
            ["lch_0.20", "lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10"],
            ["lc_0.20", "lc_0.16", "lc_0.12"],
        ],
        how_many_to_measure=2,
        multidie=True,
    ),

    # -----------------------------------------------------------------------------

    # TEST 1: "QUICK" Minimum feature size check:
    # lch, lc sweep screening
    generate_test_sweep_config(
        name="test_quick_fet_lc_lch_check",
        mod_prefixes=["mod_fet_tlm_nmos"],
        how_many_to_measure=3,
    ),

    # TEST 2: "QUICK" gate-contact overlap check:
    # lch, lc, lov sweep screening
    generate_test_sweep_config(
        name="test_quick_fet_ov_check",
        mod_prefixes=["mod_fet_ov_nmos"],
        how_many_to_measure=3,
    ),

    # TEST 3: "QUICK" Fingered FET feature size check:
    # lch, lc, width, fingers sweep screen check
    generate_test_sweep_config(
        name="test_quick_fingered_check",
        mod_prefixes=["mod_fingered_fet_tlm_nmos"],
        how_many_to_measure=10,
    ),

    # -----------------------------------------------------------------------------
    # BASIC QUICK VERIFICATION TESTS OVER
    # NEXT: do full wafer LC/LCH min feature yield check
    # -----------------------------------------------------------------------------

    # TODO: TEST 4: Full wafer LC/LCH screen
    generate_test_sweep_config(
        name="test_multidie_quick_lc_lch_check",
        mod_prefixes=["mod_fet_tlm_nmos"],
        # contains=[
        #     ["lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10", "lch_0.08", "lch_0.07", "lch_0.06"],
        #     ["lc_0.16", "lc_0.12", "lc_0.08"],
        # ],
        contains=[
            ["lch_0.20", "lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10"],
            ["lc_0.20", "lc_0.16", "lc_0.12"],
        ],
        how_many_to_measure=20,
        multidie=True,
    ),

    # -----------------------------------------------------------------------------
    # FULL WAFER YIELD CHECK DONE
    # NEXT: begin detailed wafer + die measurements
    # -----------------------------------------------------------------------------

    # TEST: Multi die, lc lch idvg/idvd sweeps at two contacts
    # Goal: DIBL, SS, channel scaling
    # across wafers: see impact of EOT
    generate_test_sweep_config(
        name="test_multidie_lc_lch_detailed_check",
        mod_prefixes=["mod_fet_tlm_nmos"],
        # contains=[
        #     ["lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10", "lch_0.08", "lch_0.07", "lch_0.06"],
        #     ["lc_0.16", "lc_0.12", "lc_0.08"],
        # ],
        contains=[
            ["lch_0.40", "lch_0.20", "lch_0.16", "lch_0.12"],
            ["lc_0.40", "lc_0.16"],
        ],
        how_many_to_measure=20,
        multidie=True,
    ),

    # TEST: Multi die, overlap sweep idvg/idvd sweeps
    # Goal: DIBL, SS, channel scaling
    # across wafers: see impact of EOT
    generate_test_sweep_config(
        name="test_multidie_overlap_detailed_check",
        mod_prefixes=["mod_fet_ov_nmos"],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.16"],
        ],
        how_many_to_measure=20,
        multidie=True,
    ),

    generate_test_sweep_config(
        name="test_asym_detailed_check",
        mod_prefixes=["mod_fet_asym_nmos"],
        # contains=[
        #     ["lch_0.16", "lch_0.14", "lch_0.12", "lch_0.10", "lch_0.08", "lch_0.07", "lch_0.06"],
        #     ["lc_0.16", "lc_0.12", "lc_0.08"],
        # ],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.16"],
        ],
        how_many_to_measure=20,
        multidie=True,
    ),

    generate_test_sweep_config(
        name="test_asym_zero_detailed_check",
        mod_prefixes=["mod_fet_ov_nmos"],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.16"],
            ["lov_0.06", "lov_0.04", "lov_0.02"],
        ],
        how_many_to_measure=20,
        multidie=True,
    ),

    # TEST X: TLM LC/LCH sweep (full)
    generate_test_sweep_config(
        name="test_full_lc_lch_sweep",
        # mod_prefixes=["mod_fet_tlm_nmos", "mod_fet_tlm_pmos"],
        mod_prefixes=["mod_fet_tlm_nmos"],
        how_many_to_measure=24,
        multidie=True,
    ),

    # TEST X: Width sweep (full)
    generate_test_sweep_config(
        name="test_quick_width_sweep",
        mod_prefixes=["mod_fet_wsweep_nmos"],
        contains=[
            ["lch_0.16"],
            ["lc_0.16"],
        ],
        how_many_to_measure=8,
    ),

    # TEST X: Width sweep (full)
    generate_test_sweep_config(
        name="test_full_width_sweep",
        mod_prefixes=["mod_fet_wsweep_nmos"],
        how_many_to_measure=8,
    ),

    # TEST X: fingered sweep (full)
    generate_test_sweep_config(
        name="test_fingered_lc_lch_sweep",
        mod_prefixes=["mod_fingered_fet_tlm_nmos_nf_4"],
        contains=[
            ["lch_0.40", "lch_0.20", "lch_0.16", "lch_0.12"],
            ["lc_0.40", "lc_0.26"],
        ],
        how_many_to_measure=12,
    ),

    ### FINAL TESTS
    generate_test_sweep_config(
        name="final_width_sweep",
        mod_prefixes=["mod_fet_wsweep_nmos", "mod_fet_wsweep_pmos"],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.20", "lc_0.16"],
        ],
        how_many_to_measure=8,
    ),

    generate_test_sweep_config(
        name="final_overlap_sweep",
        mod_prefixes=["mod_fet_ov_pmos"],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.20", "lc_0.16"],
        ],
        how_many_to_measure=24,
    ),

    generate_test_sweep_config(
        name="final_asym_sweep",
        mod_prefixes=["mod_fet_asym_pmos"],
        contains=[
            ["lch_0.16", "lch_0.12"],
            ["lc_0.20", "lc_0.16"],
        ],
        how_many_to_measure=24,
    ),

    generate_test_sweep_config(
        name="final_asym_zero_sweep",
        mod_prefixes=["mod_fet_ov_pmos"],
        contains=[
            ["lc_0.20",  "lc_0.16"],
            ["lch_0.16", "lch_0.12"],
            ["lov_0.06", "lov_0.04", "lov_0.02"],
        ],
        how_many_to_measure=24,
    ),

    generate_test_sweep_config(
        name="final_tlm_sweep",
        mod_prefixes=["mod_fet_tlm_nmos", "mod_fet_tlm_pmos"],
        contains=[
            ["lc_0.40", "lc_0.20", "lc_0.16"],
            ["lch_0.40", "lch_0.20", "lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=24,
    ),

    generate_test_sweep_config(
        name="final_tlm_idvd_sweep",
        mod_prefixes=["mod_fet_tlm_pmos"],
        contains=[
            ["lc_0.40", "lc_0.20", "lc_0.16"],
            ["lch_0.40", "lch_0.20", "lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=24,
    ),

    generate_test_sweep_config(
        name="final_fingered_compare",
        mod_prefixes=["mod_fingered_w_sweep_nmos", "mod_fingered_w_sweep_pmos"],
        contains=[
            ["nf_4", "nf_8"],
            ["w_1."],
            ["lc_0.28", "lc_0.26", "lc_0.24"],
            ["lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=8,
    ),

    generate_test_sweep_config(
        name="final_fingered_compare_nf_4",
        mod_prefixes=["mod_fingered_w_sweep_nmos", "mod_fingered_w_sweep_pmos"],
        contains=[
            ["nf_4"],
            ["w_1."],
            ["lc_0.28", "lc_0.26", "lc_0.24"],
            ["lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=8,
    ),

    generate_test_sweep_config(
        name="final_fingered_compare_nf_8",
        mod_prefixes=["mod_fingered_w_sweep_nmos", "mod_fingered_w_sweep_pmos"],
        contains=[
            ["nf_8"],
            ["w_1."],
            ["lc_0.28", "lc_0.26", "lc_0.24"],
            ["lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=8,
    ),

    generate_test_sweep_config(
        name="final_dummy",
        mod_prefixes=[
            "mod_dummy_fet_tlm_nmos", "mod_dummy_fet_tlm_pmos",
            "mod_dummy_fet_ov_nmos", "mod_dummy_fet_ov_pmos",
        ],
        contains=[
            ["lc_0.40", "lc_0.20", "lc_0.16"],
            ["lch_0.40", "lch_0.20", "lch_0.16", "lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=1,
    ),

    # for finding highest on/off and reach 100 uA/um with 100 nA/um
    generate_test_sweep_config(
        name="high_onoff_sweep",
        mod_prefixes=["mod_fet_tlm_nmos", "mod_fet_tlm_pmos"],
        contains=[
            ["lc_0.40", "lc_0.20"],
            ["lch_0.14", "lch_0.12"],
        ],
        how_many_to_measure=24,
    ),
]

# VERIFY test toml scripts
# 1. load each test toml and verify module name exists in modules list
with open(PATH_GAX9000_FET_MODULES_TOML, "rb") as f:
    all_modules = tomli.load(f)

for test_name in tests:
    print(f"Verifying: {test_name}", end="")
    path_test_toml = os.path.join(PATH_OUT, test_name + ".toml")
    with open(path_test_toml, "rb") as f:
        test_toml = tomli.load(f)
        for module_name in test_toml["sweep"]["modules"]:
            # print(f"Verifying module: {module_name}")
            assert module_name in all_modules["modules"], f"module {module_name} not found in modules list"
    print("...OK")
