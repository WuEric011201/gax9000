import os
import logging
import gevent
from controller.sweeps import MeasurementSweep
from controller.util import timestamp
from controller.sweeps.modules import load_modules_from_toml, load_sweep_from_toml
from controller.sweeps.multi_die_array import create_die_height_offset_interp2d

class SweepMultiDieModules(MeasurementSweep):
    """Implement an array sweep on multiple dies: foreach die
    coordinate, run an array sweep.
    """
    
    name = "multi_die_modules"
    
    def __repr__(self) -> str:
        return "SweepMultiDieModules"

    def __str__(self) -> str:
        return self.__repr__()
    
    def default_config_string():
        """Return default `sweep_config` argument in `run` as a dict."""
        return """
            ### list of die coordinates to sweep
            dies = [
                [0, 0],
            ]

            ### uncomment to enable per-die height compensation
            ### adjusts probe height based on interpolated measured wafer heightmap
            # height_compensation_file = "height_offset.toml"
            
            ### uncomment to use modules file .toml
            # modules_file = "modules.toml"
            ### uncomment to use sweep file .toml
            # sweep_file = "sweeps.toml"

            ### or define local modules here (same format as modules file)
            [modules]
            module1 = { name = "module1", x = 0, y = 0 }
            module2 = { name = "module2", x = 50, y = 0 }
            module3 = { name = "module3", x = 100, y = 0 }

            [sweep]
            modules = ["module1", "module2", "module3"]
        """
    
    def run(
        user,
        sweep_config,
        sweep_config_string,
        sweep_save_data,
        initial_die_x,
        initial_die_y,
        die_dx,
        die_dy,
        initial_device_row,
        initial_device_col,
        device_dx,
        device_dy,
        data_folder,
        programs,
        instr_b1500=None,
        instr_cascade=None,
        monitor_channel=None,
        signal_cancel=None,
    ):
        """Run the sweep."""

        # unpack config
        die_coordinates = sweep_config["dies"]

        if "height_compensation_file" in sweep_config:
            path_height_compensation = sweep_config["height_compensation_file"]
            die_dz_interp2d = create_die_height_offset_interp2d(path_height_compensation)
            use_height_compensation = True
        else:
            use_height_compensation = False
        
        if sweep_config.get("modules_file") is not None:
            modules = load_modules_from_toml(sweep_config["modules_file"])
        else:
            modules = sweep_config["modules"]
        
        if sweep_config.get("sweep_file") is not None:
            sweep = load_sweep_from_toml(sweep_config["sweep_file"])["modules"]
        else:
            sweep = sweep_config["sweep"]["modules"]
        
        # create closure here to simplify passing arguments
        def run_inner(
            die_x: int,
            die_y: int,
            module_str: str,
        ):
            """Run measurement at a die coordinate (die_x, die_y) for a named module device.
            Inputs:
            - `die_x`: die x coordinate.
            - `die_y`: die y coordinate.
            - `module_str`: module name (mainly as metadata).
            """
            t_measurement = timestamp()
            save_dir = os.path.join(f"die_x_{die_x}_y_{die_y}", f"gax_{module_str}_{t_measurement}")

            sweep_metadata = MeasurementSweep.save_metadata(
                user=user,
                sweep_name=SweepMultiDieModules.name,
                sweep_config_string=sweep_config_string,
                initial_die_x=initial_die_x,
                initial_die_y=initial_die_y,
                die_dx=die_dx,
                die_dy=die_dy,
                initial_device_row=initial_device_row,
                initial_device_col=initial_device_col,
                device_dx=device_dx,
                device_dy=device_dy,
                data_folder=data_folder,
                save_dir=save_dir,
                save_data=sweep_save_data,
                programs=programs,
            )

            for pr in programs:
                logging.info(f"[die=({die_x},{die_y}), module={module_str}] Running {pr.name}...")
                MeasurementSweep.run_single(
                    instr_b1500=instr_b1500,
                    monitor_channel=monitor_channel,
                    signal_cancel=signal_cancel,
                    sweep_metadata=sweep_metadata,
                    data_folder=data_folder,
                    save_dir=save_dir,
                    save_data=sweep_save_data,
                    program=pr,
                )

                # yields thread for other tasks (so data gets pushed)
                # TODO: proper multithreaded task
                gevent.sleep(0.3)
            
            # yields thread for other tasks
            # TODO: proper multithreaded task
            gevent.sleep(0.3)

        # store current die location
        current_die_x = initial_die_x
        current_die_y = initial_die_y
        instr_cascade.move_chuck_home()  # Tong ADDED
        for die_coord in die_coordinates:
            die_x, die_y = die_coord
            
            # make sure die coords are integers
            die_x = int(die_x)
            die_y = int(die_y)

            logging.info(f"Moving to die ({die_x}, {die_y})")

            # get height for contact
            if use_height_compensation:
                dz = die_dz_interp2d(die_x, die_y)
            else:
                dz = 0
            
            if instr_cascade is not None:
                # move to die location and set home origin
                if current_die_x != die_x or current_die_y != die_y:
                    # move to contact height (stop contacting devices)
                    instr_cascade.move_contacts_up()
                    gevent.sleep(0.5) # ensure small delay

                    # move chuck to target die location using relative coord from current die
                    dx_to_die = (die_x - current_die_x) * die_dx
                    dy_to_die = (die_y - current_die_y) * die_dy
                    print(f"die_dx={die_dx}, die_dy={die_dy}, dx_to_die={dx_to_die}, dy_to_die={dy_to_die}")
                    logging.info(f"Tong Moving to die ({die_x}, {die_y}) from ({current_die_x},{current_die_y}) ")
                    # instr_cascade.move_chuck_relative_to_home(x=dx_to_die, y=dy_to_die)  <-----------Tong: Problematic line
                    instr_cascade.write(f"MoveChuck {dx_to_die} {dy_to_die} H Y 100")
                    instr_cascade.read() # read required to flush response
                    instr_cascade.query("*OPC?")

                    # TODO: do chuck height compensation from baseline

                    # update to new die location and set check home position
                    # (for array measurement) to current die location
                    instr_cascade.set_chuck_home()
                    current_die_x = die_x
                    current_die_y = die_y

            # move to each module and probe
            for module_name in sweep:
                module = modules[module_name]

                # local (x, y) coordinates relative to die home
                x_module = module["x"]
                y_module = module["y"]
                
                if instr_cascade is not None:
                    instr_cascade.move_contacts_up()
                    instr_cascade.move_chuck_relative_to_home(x=x_module, y=y_module)
                    instr_cascade.move_to_contact_height_with_offset(dz)
                
                run_inner(
                    die_x=die_x,
                    die_y=die_y,
                    module_str=module_name,
                )

                # check cancel signal and return if received
                if signal_cancel is not None and signal_cancel.is_cancelled():
                    logging.info("Measurement cancelled by signal.")
                    return

