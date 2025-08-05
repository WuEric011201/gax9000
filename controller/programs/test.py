import logging
import traceback
import numpy as np
import gevent
import pyvisa
import logging
from time import time
from tabulate import tabulate
from controller.sse import EventChannel
from controller.programs import MeasurementProgram, MeasurementResult, SweepType
from controller.util import into_sweep_range, parse_keysight_str_values, iter_chunks, map_smu_to_slot, exp_moving_avg_with_init, dict_np_array_to_json_array
import os, json
from controller.backend import ControllerSettings

rm = pyvisa.ResourceManager()
print(rm.list_resources())
path_config = os.path.join("settings", "config.json")
if os.path.exists(path_config):
    with open(path_config, "r") as f:
        config = ControllerSettings(**json.load(f))
else:
    config = ControllerSettings() # default

print(f"CONFIG = {config}")

instr_b1500 = rm.open_resource(
    f"GPIB0::{config.gpib_b1500}::INSTR",
    # read_termination="\n", # default, not needed
    # write_termination="\n",
)

print(instr_b1500.query("*IDN?"))
instr_b1500.write("*RST")
