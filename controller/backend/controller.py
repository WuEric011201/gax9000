"""
Controller API

Handle read/write wafer and controller config from client ui.
"""
import time
from typing import Callable
import os
import logging
import traceback
import json
import tomli
import gevent
from gevent.lock import BoundedSemaphore
import pyvisa
from flask_restful import Api, Resource, reqparse
from controller.sse import EventChannel
from controller.programs import MEASUREMENT_PROGRAMS, MeasurementProgram
from controller.sweeps import MEASUREMENT_SWEEPS, MeasurementSweep, RunMeasurementProgram
from controller.util import SignalCancelTask
from controller.util.string import strip_leading_whitespace


class UserGlobalSettings():
    """Saved user global config settings."""
    def __init__(
        self,
        username,
        die_size_x=10000,
        die_size_y=10000,
        die_offset_x=0,
        die_offset_y=0,
        current_die_x=0,
        current_die_y=0,
        device_x=0,
        device_y=0,
        device_row=0,
        device_col=0,
        data_folder="",
    ):
        self.username = username
        self.die_size_x = die_size_x
        self.die_size_y = die_size_y
        self.die_offset_x = die_offset_x
        self.die_offset_y = die_offset_y
        self.current_die_x = current_die_x
        self.current_die_y = current_die_y
        self.device_x = device_x
        self.device_y = device_y
        self.device_row = device_row
        self.device_col = device_col
        self.data_folder = data_folder
    
    def default(username):
        """Default settings."""
        return UserGlobalSettings(
            username=username,
        )


class UserProfile():
    """Contain all user settings. Used as an intermediate cache
    before periodically saving dirty settings to disk."""
    def __init__(
        self,
        global_settings,
        program_settings = {},
        measurement_settings = {}
    ):
        self.global_settings = global_settings
        self.program_settings = program_settings
        self.measurement_settings = measurement_settings
        self.dirty_global_settings = False
        self.dirty_program_settings = set()     # set of dirty program name strings
        self.dirty_measurement_settings = set() # set of dirty measurement sweep name strings
    
    def default(username):
        """Create user profile with default settings."""
        return UserProfile(
            global_settings=UserGlobalSettings.default(username),
            program_settings={}, # TODO
            measurement_settings={}, # TODO
        )
    
    def save(self, path):
        """Save settings to path.
        Inputs:
        - path: Folder containing all individual user folders, e.g. "settings/users/".
        Returns:
        - True if any settings were dirty and saved, False if not.
        """
        path_user = os.path.join(path, self.global_settings.username)
        os.makedirs(path_user, exist_ok=True)

        did_update = False

        path_global_settings = os.path.join(path_user, "settings.json")
        path_program_settings_dir = os.path.join(path_user, "programs")
        path_measurement_settings_dir = os.path.join(path_user, "measurements")

        if self.dirty_global_settings or not os.path.exists(path_global_settings):
            with open(path_global_settings, "w+") as f:
                json.dump(self.global_settings.__dict__, f, indent=2)
            did_update = True

        # NOTE: here we only save names of program and measurement settings
        # that were dirty (as in used by the user). So these are all lazily saved.
        # If the user never uses a program, its settings will never be saved.
        
        if len(self.dirty_program_settings) > 0:
            if not os.path.exists(path_program_settings_dir):
                os.makedirs(path_program_settings_dir, exist_ok=True)
            
            for program_name, program_settings in self.program_settings.items():
                # TODO
                pass
            did_update = True
            
        if len(self.dirty_measurement_settings) > 0:
            if not os.path.exists(path_measurement_settings_dir):
                os.makedirs(path_measurement_settings_dir, exist_ok=True)
            
            for measurement_name, measurement_settings in self.measurement_settings.items():
                # TODO
                pass
            did_update = True
        
        # clear dirty flags
        self.dirty_global_settings = False
        self.dirty_program_settings = set()
        self.dirty_measurement_settings = set()
        
        return did_update

    def load(path_users, username):
        """Return new user profile settings object from data in path.
        If files are missing, this will create default settings objects.
        Inputs:
        - path_users: Folder for all users, e.g. "settings/users/"
        - username: Name of user to load settings for. User settings path
            is by joining `path_users + username`.
        """
        # derive username from directory in path
        path = os.path.join(path_users, username)

        path_global_settings = os.path.join(path, "settings.json")
        path_program_settings_dir = os.path.join(path, "programs")
        path_measurement_settings_dir = os.path.join(path, "measurements")
        
        if os.path.exists(path_global_settings):
            with open(path_global_settings, "r") as f:
                global_settings = UserGlobalSettings(**json.load(f))
        else:
            global_settings = UserGlobalSettings.default(username)
        
        return UserProfile(
            global_settings=global_settings,
            program_settings={}, # TODO
            measurement_settings={}, # TODO
        )
        
class InstrumentCascade():
    """Interface for controlling cascade auto probe station instrument
    through GPIB. Contains helper functions for common movement operations.

    Note: moving the chuck across wafer can take few seconds, make sure 
    timeout is long enough to accomodate.
    """
    def __init__(
        self,
        gpib_addr: str,                      # GPIB address of instrument
        invert_direction: bool = True,       # X,Y axis direction (inverted +x,+y is towards top right)
        base_contact_height: float = 5000.0, # height in um for contacting probes to device
        timeout: float = 8.0,                # standard instrument timeout in seconds
    ):
        # try connecting through gpib
        gpib_resource_manager = pyvisa.ResourceManager()
        self.gpib = gpib_resource_manager.open_resource(gpib_addr)
        self.gpib.timeout = timeout * 1000.0 # convert to millis
        self.gpib_timeout_millis = timeout * 1000.0

        # get instrument identification string
        self.identifier = self.gpib.query("*IDN?")
        
        # flag for inverting coordinate direction
        # inverted = True means +x,+y is towards top right
        self.invert_direction = invert_direction

        # height in um for contacting probes to device
        # used as baseline when doing stage (chuck) height compensation
        # for multi-die measurement sweeps (to account for non-flat stage
        # and wafer warping from stress)
        self.base_contact_height = base_contact_height
        
        # flag that a chuck home position has been set by any prior measurement
        # this is required to ensure `move_chuck_home` is safe
        self.chuck_home_position_set = False
    
    def close(self):
        """Close instrument GPIB connection."""
        self.gpib.close()
    
    def write(self, command: str):
        """Manually write a command to instrument (non-blocking)."""
        self.gpib.write(command)
    
    def query(self, command: str):
        """Manually write and query a command from instrument (blocking)."""
        return self.gpib.query(command)
    
    def read(self):
        """Manually read a response from instrument (blocking)."""
        return self.gpib.read()

    def set_chuck_home(self):
        """Set cascade autoprobe chuck home to current location.
        This is used in measurements to probe arrays relative to
        starting location.
            SetChuckHome Mode Unit
        Mode:
            0 - use current position
            V - use given value
        Unit
            Y - micron (default)
            I - mils
        """
        self.gpib.write(f"SetChuckHome 0 Y")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
        self.chuck_home_position_set = True
    
    def move_chuck_relative(self, dx, dy):
        """Moves cascade autoprobe chuck relative to current location
        by (dx, dy). Command format is
            MoveChuck X Y PosRef Unit Velocity Compensation
        X: dx
        Y: dy
        PosRef:
            - H - home (default)
            - Z - zero
            - C - center
            - R - current position
        Unit:
            - Y - Micron (default)
            - I - Mils
            - X - Index
            - J - jog
        Velocity: velocity in percent (100% default)
        Compensation:
            - D - default (kernel setup default compensation)
            - T - technology, use prober, offset, and tech compensation
            - O - offset, use prober and offset
            - P - prober, use only prober
            - N - none, no compensation
        """
        if self.invert_direction:
            dx_ = -dx
            dy_ = -dy
        else:
            dx_ = dx
            dy_ = dy
        
        self.gpib.write(f"MoveChuck {dx_} {dy_} R Y 100 N")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
    
    def move_chuck_relative_to_home(self, x, y, timeout=None):
        """Moves wafer chuck relative to home position. See `move_chuck_relative`
        for MoveChuck command documentation. Accept custom timeout in seconds, for
        doing large wafer movements (e.g. cross die).
        """
        if self.invert_direction:
            x_ = -x
            y_ = -y
        else:
            x_ = x
            y_ = y
        
        if timeout is not None:
            self.gpib.timeout = timeout * 1000.0
        
        self.gpib.write(f"MoveChuck {x_} {y_} H Y 100 N")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
        
        if timeout is not None: # reset
            self.gpib.timeout = self.gpib_timeout_millis
    
    def move_chuck_home(self):
        """Move chuck to previously set home position. See `move_chuck_relative`
        for MoveChuck command documentation.
        """
        if self.chuck_home_position_set:
            self.gpib.write(f"MoveChuck 0 0 H Y 100")
            self.gpib.read() # read required to flush response
            self.gpib.query("*OPC?")
        else:
            logging.error("`move_chuck_home` failed: no home set.")
    
    def move_contacts_up(self):
        """Move contacts up (internally moves chuck down). Command is
            `MoveChuckAlign Velocity` (velocity = 100% default)
        """
        self.gpib.write(f"MoveChuckAlign 50")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
    
    def move_contacts_down(self):
        """Moves contacts down to touch device (internally moves chuck up).
            `MoveChuckContact Velocity` (velocity = 100% default)
        """
        self.gpib.write(f"MoveChuckContact 50")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
    
    def move_to_contact_height_with_offset(self, dz: float):
        """Moves contacts down to touch device with some dz offset,
        for height compensation:
            `MoveChuckZ Z PosRef Unit Velocity`
            Z: z height
            PosRef: 
                - Z: zero (default)
                - H: contact
                - R: current position
            Unit:
                - Y: micron (default)
                - I: mils
                - J: job
            Velocity: velocity in percent (100 = default 100%)
            Comp. level: compensation level
                - D: default, uses one of four below
                - T: technology + prober + offset (default)
                - O: offset - use prober and offset
                - P: prober compensation only
                - N: no compensation
        """
        print(f"MOVE TO DZ {dz}")
        self.gpib.write(f"MoveChuckZ {dz} H Y 50 N")
        self.gpib.read() # read required to flush response
        self.gpib.query("*OPC?")
    

class ControllerSettings():
    """Global controller settings. These are saved each time
    value is changed.
    """
    def __init__(
        self,
        gpib_b1500: int = 17,                # gpib id of b1500 instrument
        gpib_cascade: int = 22,              # gpib id of cascade instrument
        users: list = ["public"],            # list of username strings
        invert_direction: bool = True,       # invert chuck movement directions (if true, topleft is (+x,+y))
        base_contact_height: float = 5000.0, # height in um for contacting probes to device
        smu_slots: dict = {},                # map SMU # to slot #
    ):
        self.gpib_b1500 = gpib_b1500
        self.gpib_cascade = gpib_cascade
        self.users = users
        self.invert_direction = invert_direction
        self.base_contact_height = base_contact_height
        self.smu_slots = smu_slots

    def __repr__(self):
        return self.__dict__.__repr__()

    def default():
        """Return a default settings object."""
        return ControllerSettings(
            gpib_b1500=17,
            gpib_cascade=22,
            users=["public"],
            invert_direction=True,
            base_contact_height=5000.0,
            smu_slots={},
        )

    
class Controller():
    def __init__(
        self,
        path_settings: str,
        path_users: str,
        monitor_channel: EventChannel,
    ):
        """Singleton controller for managing instrument resources.
        """
        # py visa resource manager
        self.resource_manager = pyvisa.ResourceManager()
        # b1500 parameter analyzer instrument
        self.instrument_b1500 = None
        # cascade instrument
        self.instrument_cascade: InstrumentCascade = None
        # channel to broadcast measurement data results
        self.monitor_channel = monitor_channel
        # controller global settings
        self.path_settings = path_settings
        # do initial controller settings load
        self.load_settings()
        # controller user settings path
        self.path_users = path_users
        # user settings, maps username: str => UserProfile: class
        self.users = {}
        # repeating task to save user settings
        self.task_save_user_settings = gevent.spawn(self._task_save_user_settings)
        # current main instrument task, this must be locked and synchronized
        # to ensure instrument is only running single task at a time
        self.task = None
        # lock on instrument task
        self.task_lock = BoundedSemaphore(value=1)
        # cancel task signal
        self.signal_cancel_task = SignalCancelTask()
    
    def connect_b1500(self, gpib: int):
        """Connect to b1500 instrument resource through GPIB
        and return identification string."""
        addr = f"GPIB0::{gpib}::INSTR" # TODO: address string should be setting
        self.instrument_b1500 = self.resource_manager.open_resource(addr)
        return self.instrument_b1500.query("*IDN?")

    def disconnect_b1500(self):
        """Disconnect from b1500 instrument."""
        if self.instrument_b1500 is not None:
            # check if task lock active (do not allow disconnecting
            # in middle of measurement)
            if self.task_lock.acquire(blocking=False, timeout=None):
                self.instrument_b1500.close()
                self.instrument_b1500 = None
                self.task_lock.release()

    def connect_cascade(self, gpib: int):
        """Connect to cascade instrument resource through GPIB
        and return identification string."""
        addr = f"GPIB0::{gpib}::INSTR" # TODO: address string should be setting
        try:
            self.instrument_cascade = InstrumentCascade(
                gpib_addr=addr,
                invert_direction=self.settings.invert_direction,
                base_contact_height=self.settings.base_contact_height,
            )
            return self.instrument_cascade.identifier
        except:
            self.instrument_cascade = None
            logging.error(f"Failed to connect to cascade instrument at {addr}")
            return f"FAILED connection to {addr}"

    def disconnect_cascade(self):
        """Disconnect from cascade instrument."""
        if self.instrument_cascade is not None:
            self.instrument_cascade.close()
            self.instrument_cascade = None
    
    def load_settings(self):
        """Load controller settings from file."""
        with open(self.path_settings, "r") as f:
            self.settings = ControllerSettings(**json.load(f))
    
    def save_settings(self):
        """Save controller settings to file."""
        with open(self.path_settings, "w+") as f:
            json.dump(self.settings.__dict__, f, indent=2)

    def get_user_settings(self, username):
        """Get user settings.
        If it doesn't exist, load data from saved user `settings.json`.
        If saved settings json file does not exist, save default user
        settings first.
        """
        if username not in self.users:
            path_user = os.path.join(self.path_users, username)
            if os.path.exists(path_user):
                self.users[username] = UserProfile.load(self.path_users, username)
            else: # generate default user settings
                logging.info(f"Creating default user settings for {username} at: {path_user}")
                self.users[username] = UserProfile.default(username)
                self.users[username].save(self.path_users)
        
        return self.users[username].global_settings

    def save_user_settings(self):
        """Saves dirty user settings to .json files storage."""
        for username, user in self.users.items():
            if user.save(self.path_users):
                logging.info(f"Saved user settings: {username}")
    
    def _task_save_user_settings(self):
        """Internal task that runs in gevent greenlet to periodically
        save dirty user settings."""
        while True:
            self.save_user_settings()
            gevent.sleep(10.0) # currently hardcoded save every 10s
    
    def set_user_setting(self, user: str, setting: str, value):
        """Set user global setting.
        This will mark the user settings as dirty.
        """
        if user in self.users:
            if hasattr(self.users[user].global_settings, setting):
                self.users[user].global_settings.__setattr__(setting, value)
                self.users[user].dirty_global_settings = True
            else:
                logging.warn(f"set_user_setting() Invalid setting: {setting}")
        else:
            logging.warn(f"set_user_setting() Invalid user: {user}")
    
    def get_measurement_program_config_string(
        self,
        username: str,
        program: str,
    ) -> str:
        """Get measurement program config string for user and program.
        Returns either user's current program config, or generates new
        default config for the program.
        """
        print(username, program)
        if username in self.users:
            # get/create program config path if it does not exist
            path_user_programs = os.path.join(self.path_users, username, "program")
            if not os.path.exists(path_user_programs):
                os.makedirs(path_user_programs)
            
            path_program = os.path.join(path_user_programs, program + ".toml")
            if os.path.exists(path_program):
                with open(path_program, "r") as f:
                    return f.read()
            else: # generate default user settings
                logging.info(f"Creating default program {program}.toml config for {username} at: {path_program}")
                config_str = MeasurementProgram.get(program).default_config_string()
                config_str = strip_leading_whitespace(config_str) # strip leading whitespace on all lines
                with open(path_program, "w+") as f:
                    f.write(config_str)
                return config_str

        return None

    def get_measurement_program_config(
        self,
        username: str,
        program: str,
    ) -> dict:
        """Get measurement program config dict for user and program as dict.
        Re-uses get_measurement_program_config_string() to get the config string,
        then parses it as a toml.
        """
        config_str = self.get_measurement_program_config_string(username, program)
        if config_str is not None:
            return tomli.loads(config_str)

    def set_measurement_program_config(
        self,
        username: str,
        program: str,
        config_str: str,
    ):
        """Set measurement program config for user and program."""
        print(username, program, config_str)
        if username in self.users:
            # get/create program config path if it does not exist
            path_user_programs = os.path.join(self.path_users, username, "program")
            if not os.path.exists(path_user_programs):
                os.makedirs(path_user_programs)
            
            path_program = os.path.join(path_user_programs, program + ".toml")
            with open(path_program, "w+") as f:
                f.write(config_str)
    
    def get_measurement_sweep_config_string(
        self,
        username: str,
        sweep: str,
    ) -> str:
        """Get measurement sweep config for user and program.
        Returns either user's current sweep config, or generates new
        default config for the sweep.
        """
        if username in self.users:
            # get/create program config path if it does not exist
            path_user_sweeps = os.path.join(self.path_users, username, "sweep")
            if not os.path.exists(path_user_sweeps):
                os.makedirs(path_user_sweeps)
            
            path_sweep = os.path.join(path_user_sweeps, sweep + ".toml")
            if os.path.exists(path_sweep):
                with open(path_sweep, "r") as f:
                    return f.read()
            else: # generate default user settings
                logging.info(f"Creating default sweep {sweep}.toml config for {username} at: {path_sweep}")
                config_str = MeasurementSweep.get(sweep).default_config_string()
                config_str = strip_leading_whitespace(config_str) # strip leading whitespace on all lines
                with open(path_sweep, "w+") as f:
                    f.write(config_str)
                return config_str

        return None

    def get_measurement_sweep_config(
        self,
        username: str,
        sweep: str,
    ) -> dict:
        """Get measurement sweep config for user and program.
        Returns either user's current sweep config, or generates new
        default config for the sweep.
        """
        config_str = self.get_measurement_sweep_config_string(username, sweep)
        if config_str is not None:
            return tomli.loads(config_str)
    
    def set_measurement_sweep_config(self, username, sweep, config):
        """Set measurement program config for user and program."""
        print(username, sweep, config)
        if username in self.users:
            # get/create program config path if it does not exist
            path_user_sweeps = os.path.join(self.path_users, username, "sweep")
            if not os.path.exists(path_user_sweeps):
                os.makedirs(path_user_sweeps)
            
            path_sweep = os.path.join(path_user_sweeps, sweep + ".toml")
            with open(path_sweep, "w+") as f:
                if isinstance(config, str):
                    f.write(config)
                else:
                    f.write(str(config))
        
    def run_measurement(
        self,
        user: str,
        initial_die_x: int,
        initial_die_y: int,
        die_dx: float,
        die_dy: float,
        initial_device_row: int,
        initial_device_col: int,
        device_dx: float,
        device_dy: float,
        data_folder: str,
        programs: list[RunMeasurementProgram],
        sweep: MeasurementSweep,
        sweep_config: dict,
        sweep_config_string: str,
        sweep_save_data: bool,
        sweep_save_image: bool,
        callback: Callable,
    ):
        print("RUNNING MEASUREMENT")
        print("user =", user)
        print("initial_die_x =", initial_die_x)
        print("initial_die_y =", initial_die_y)
        print("die_dx =", die_dx)
        print("die_dy =", die_dy)
        print("initial_device_row =", initial_device_row)
        print("initial_device_col =", initial_device_col)
        print("device_dx =", device_dx)
        print("device_dy =", device_dy)
        print("data_folder =", data_folder)
        print("programs =", programs)
        print("sweep =", sweep)
        print("sweep_config =", sweep_config)
        print("sweep_config_string =", sweep_config_string)
        print("sweep_save_data =", sweep_save_data)
        print("sweep_save_image =", sweep_save_image)

        # verify data folder exists
        if sweep_save_data and not os.path.exists(data_folder):
            logging.error(f"Data folder {data_folder} does not exist. Cancelling measurement sweep.")
            callback(False)
            return
        
        # inject global settings into program configs:
        # - smu slot settings
        if len(self.settings.smu_slots) > 0:
            for pr in programs:
                pr.config["smu_slots"] = self.settings.smu_slots
    
        # try acquire instrument task lock
        if self.task_lock.acquire(blocking=False, timeout=None):
            
            # reset cancel task signal
            self.signal_cancel_task.reset()

            # set current probe station chuck home position
            if self.instrument_cascade is not None:
                self.instrument_cascade.set_chuck_home()
            
            def task():
                status = False # measurement result status: TODO replace with something better

                logging.info(f"Beginning measurement sweep: {sweep}")

                try:
                    sweep.run(
                        instr_b1500=self.instrument_b1500,
                        instr_cascade=self.instrument_cascade,
                        user=user,
                        sweep_config=sweep_config,
                        sweep_config_string=sweep_config_string,
                        sweep_save_data=sweep_save_data,
                        die_dx=die_dx,
                        die_dy=die_dy,
                        initial_die_x=initial_die_x,
                        initial_die_y=initial_die_y,
                        device_dx=device_dx,
                        device_dy=device_dy,
                        initial_device_row=initial_device_row,
                        initial_device_col=initial_device_col,
                        data_folder=data_folder,
                        programs=programs,
                        monitor_channel=self.monitor_channel,
                        signal_cancel=self.signal_cancel_task,
                    )
                    status = True # successfully finished
                except Exception as err:
                    logging.error(f"Measurement FAILED: {err}")
                    logging.error(traceback.format_exc())
                    if self.instrument_b1500 is not None:
                        self.instrument_b1500.write("DZ") # ensure channels are zero-d if measurement ran into error
                
                # measurement ended (finished or cancelled)
                if self.instrument_b1500 is not None:
                    logging.info("Measurement finished, turning off SMUs with 'CL' signal")
                    self.instrument_b1500.write("CL")

                # clear task lock and reset cancel task signal
                self.task_lock.release()
                self.signal_cancel_task.reset()

                logging.info(f"Finished measurement sweep")
                callback(status)

            self.task = gevent.spawn(task)
            return

        logging.error(f"Failed to start measurement lock: Another task is already running")
        callback(False)
        return
    
    def cancel_measurement(self):
        """Acquire cancel task lock and raise signal to stop measuring.
        This will stop the measurement after the current program step finishes.
        """
        self.signal_cancel_task.cancel()


class ControllerApiHandler(Resource):
    def __init__(
        self,
        channel: EventChannel,
        monitor_channel: EventChannel,
        controller: Controller,
    ):
        # main SSE event channel for pushing data responses to frontend
        self.channel = channel
        # SSE event channel for push data responses to monitoring frontend
        self.monitor_channel = monitor_channel
        # instrument controller class
        self.controller = controller
        # put request handlers
        self.put_handlers = {
            "run_measurement": self.run_measurement,
            "cancel_measurement": self.cancel_measurement,
            "connect_b1500": self.connect_b1500,
            "disconnect_b1500": self.disconnect_b1500,
            "set_b1500_gpib_address": self.set_b1500_gpib_address,
            "connect_cascade": self.connect_cascade,
            "disconnect_cascade": self.disconnect_cascade,
            "get_user_settings": self.get_user_settings,
            "set_user_setting": self.set_user_setting,
            "get_measurement_program_config": self.get_measurement_program_config,
            "set_measurement_program_config": self.set_measurement_program_config,
            "get_measurement_sweep_config": self.get_measurement_sweep_config,
            "set_measurement_sweep_config": self.set_measurement_sweep_config,
            "move_chuck_relative": self.move_chuck_relative,
            "move_chuck_home": self.move_chuck_home,
            "move_contacts_up": self.move_contacts_up,
            "move_contacts_down": self.move_contacts_down,
        }
    
    def run_measurement(
        self,
        user: str,
        initial_die_x: int,
        initial_die_y: int,
        die_dx: float,
        die_dy: float,
        initial_device_row: int,
        initial_device_col: int,
        device_dx: float,
        device_dy: float,
        data_folder: str,
        programs: list[str],
        program_configs: list[str],
        sweep: str,
        sweep_config: str,
        sweep_save_data: bool,
        sweep_save_image: bool,
    ):
        """Run measurement task."""
        print("BEGIN MEASUREMENT PARSING")
        print("user =", user)
        print("initial_die_x =", initial_die_x)
        print("initial_die_y =", initial_die_y)
        print("die_dx =", die_dx)
        print("die_dy =", die_dy)
        print("initial_device_row =", initial_device_row)
        print("initial_device_col =", initial_device_col)
        print("device_dx =", device_dx, type(device_dx))
        print("device_dy =", device_dy, type(device_dy))
        print("data_folder =", data_folder)
        print("programs =", programs)
        print("program_configs =", program_configs)
        print("sweep =", sweep)
        print("sweep_config =", sweep_config)
        print("sweep_save_data =", sweep_save_data)

        # make sure number of programs and program configs is the same
        if len(programs) != len(program_configs):
            logging.error("Number of programs and program configs is not the same")
            return self.signal_measurement_failed("Number of programs and program configs is not the same")

        # parse program configs
        run_programs: list[RunMeasurementProgram] = []
        for pr, pr_config in zip(programs, program_configs):
            instr_program = MeasurementProgram.get(pr)
            if instr_program is None:
                logging.error(f"Invalid program: {pr}")
                return self.signal_measurement_failed(f"Invalid program: {pr}")
            
            try:
                program_config_dict = tomli.loads(pr_config)
            except Exception as err:
                logging.error(f"Invalid program config: {err}")
                return self.signal_measurement_failed(f"Invalid program config for {pr}: {pr_config}")
            
            run_programs.append(RunMeasurementProgram(
                program = instr_program,
                config = program_config_dict,
                config_string = pr_config,
            ))
        
        # get sweep and parse sweep config
        instr_sweep = MeasurementSweep.get(sweep)
        if instr_sweep is None:
            logging.error(f"Invalid sweep type: {sweep}")
            return self.signal_measurement_failed(f"Invalid sweep: {sweep}")
        try:
            sweep_config_dict = tomli.loads(sweep_config)
        except Exception as err:
            logging.error(f"Invalid sweep config: {err}")
            return self.signal_measurement_failed("Invalid sweep config")
        
        # passed program and sweep config validation/parsing,
        # save sweep config and program configs to disk to cache settings
        self.set_measurement_sweep_config(user, instr_sweep.name, sweep_config)
        for pr in run_programs:
            self.set_measurement_program_config(user, pr.name, pr.config_string)
        
        # run internal measurement task
        self.controller.run_measurement(
            user=user,
            initial_die_x=initial_die_x,
            initial_die_y=initial_die_y,
            die_dx=die_dx,
            die_dy=die_dy,
            initial_device_row=initial_device_row,
            initial_device_col=initial_device_col,
            device_dx=device_dx,
            device_dy=device_dy,
            data_folder=data_folder,
            programs=run_programs,
            sweep=instr_sweep,
            sweep_config=sweep_config_dict,
            sweep_config_string=sweep_config,
            sweep_save_data=sweep_save_data,
            sweep_save_image=sweep_save_image,
            callback=self.signal_measurement_finished,
        )
    
    def cancel_measurement(self):
        """Cancel measurement task."""
        self.controller.cancel_measurement()
    
    def signal_measurement_finished(
        self,
        success,
    ):
        # TODO: better status response about measurement
        if success:
            status = "success"
        else:
            status = "error"
        
        self.channel.publish({
            "msg": "measurement_finish",
            "data": {
                "status": status,
            },
        })

    def signal_measurement_failed(
        self,
        error,
    ):
        self.channel.publish({
            "msg": "measurement_error",
            "data": {
                "error": error,
            },
        })

    def connect_b1500(self, gpib_address):
        idn = self.controller.connect_b1500(gpib_address)
        logging.info(f"Connected (GPIB: {gpib_address}): {idn}")
        self.channel.publish({
            "msg": "connect_b1500_idn",
            "data": {
                "idn": idn,
            },
        })

    def disconnect_b1500(self):
        self.controller.disconnect_b1500()
        self.channel.publish({
            "msg": "disconnect_b1500",
            "data": {},
        })

    def set_b1500_gpib_address(self, gpib_address):
        """Set B1500 GPIB address setting."""
        self.controller.settings.gpib_b1500 = gpib_address
        self.controller.save_settings()
    
    def connect_cascade(self, gpib_address):
        idn = self.controller.connect_cascade(gpib_address)
        logging.info(f"Connected (GPIB: {gpib_address}): {idn}")
        self.channel.publish({
            "msg": "connect_cascade_idn",
            "data": {
                "idn": idn,
            },
        })
    
    def disconnect_cascade(self):
        self.controller.disconnect_cascade()
        self.channel.publish({
            "msg": "disconnect_cascade",
            "data": {},
        })

    def set_cascade_gpib_address(self, gpib_address):
        """Set Cascade GPIB address setting."""
        self.controller.settings.gpib_cascade = gpib_address
        self.controller.save_settings()
    
    def get_user_settings(self, user):
        """Get user settings."""
        user_settings = self.controller.get_user_settings(user)
        self.channel.publish({
            "msg": "set_user_settings",
            "data": {
                "settings": user_settings.__dict__,
            },
        })
    
    def set_user_setting(
        self,
        user: str,
        setting: str,
        value,
    ):
        """Update a user setting to new value."""
        self.controller.set_user_setting(user, setting, value)
    
    def get_measurement_program_config(
        self,
        user: str,
        program: str,
        index: int,
    ):
        """Get measurement program config for user and program."""
        config_str = self.controller.get_measurement_program_config_string(user, program)
        if config_str is not None:
            self.channel.publish({
                "msg": "measurement_program_config",
                "data": {
                    "name": program,
                    "index": index,
                    "config": config_str,
                },
            })

    def set_measurement_program_config(
        self,
        user: str,
        program: str,
        config_str: str,
    ):
        """Set measurement program config for user and program."""
        self.controller.set_measurement_program_config(user, program, config_str)

    def get_measurement_sweep_config(
        self,
        user: str,
        sweep: str,
    ):
        """Get measurement program config for user and program and
        push data back to event channel."""
        config_str = self.controller.get_measurement_sweep_config_string(user, sweep)
        if config_str is not None:
            self.channel.publish({
                "msg": "measurement_sweep_config",
                "data": {
                    "name": sweep,
                    "config": config_str,
                },
            })

    def set_measurement_sweep_config(
        self,
        user: str,
        sweep: str,
        config: str,
    ):
        """Set measurement program config for user and program."""
        self.controller.set_measurement_sweep_config(user, sweep, config)

    def move_chuck_relative(self, dx, dy):
        """Move chuck relative to current position."""
        if self.controller.instrument_cascade is not None:
            self.controller.instrument_cascade.move_chuck_relative(dx, dy)
        else:
            logging.error("`move_chuck_relative` failed: no Cascade connected.")
        
    def move_chuck_home(self):
        """Move chuck relative to current position."""
        if self.controller.instrument_cascade is not None:
            self.controller.instrument_cascade.move_chuck_home()
        else:
            logging.error("`move_chuck_home` failed: no Cascade connected.")
        
    def move_contacts_up(self):
        """Move contacts up."""
        if self.controller.instrument_cascade is not None:
            self.controller.instrument_cascade.move_contacts_up()
        else:
            logging.error("`move_contacts_up` failed: no Cascade connected.")
        
    def move_contacts_down(self):
        """Move contacts down."""
        if self.controller.instrument_cascade is not None:
            self.controller.instrument_cascade.move_contacts_down()
        else:
            logging.error("`move_contacts_down` failed: no Cascade connected.")
    
    def get(self):
        """Returns global controller config settings."""
        # reload controller settings on page load
        self.controller.load_settings()
        return {
            "gpib_b1500": self.controller.settings.gpib_b1500,
            "gpib_cascade": self.controller.settings.gpib_cascade,
            "users": self.controller.settings.users,
            "programs": MEASUREMENT_PROGRAMS,
            "sweeps": MEASUREMENT_SWEEPS,
        }

    def put(self):
        """Main handler for updating global config or user profile data.
        Put requests are all in format:
            {
                "msg": "request_event",
                "data": { ... },
            }
        where "msg" routes to the event handler and "data" contains
        the event handler function inputs.
        """
        parser = reqparse.RequestParser()
        parser.add_argument("msg", type=str)
        parser.add_argument("data", type=dict)

        try:
            args = parser.parse_args()
            logging.info(f"PUT {args}")
            if args["msg"] in self.put_handlers:
                kwargs = args["data"]
                print(kwargs)
                self.put_handlers[args["msg"]](**kwargs)
        except Exception as exception:
            logging.error(exception)
            logging.error(traceback.format_exc())
        