"""
Implement FET basic current-voltage (I-V) measurements.
                 Vg
                __|__
        Vd  ____|   |___ Vs
                  |
                 Vsub

Sweep format supports series of forward/reverse sweeps to do device
"burn-in" or to see hysteresis. Sequence is
    "frfr..."
- "f": forward sweep
- "r": reverse sweep

Standard is "fr" to do a combined forward/reverse sweep to see hysteresis.
"""

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


def measurement_keysight_b1500_setup(
    instr_b1500,
    query_error,
    probe_gate: int,
    probe_source: int,
    probe_drain: int,
    probe_sub: int,
    id_compliance: float,
    ig_compliance: float,
    pow_compliance: float,
    pulsed: bool = False, # flag for dc pulsed, 500 us pulse widths
    pulse_width: float = 0.0005, # pulse width in secs
    pulse_period: float = 0.010, # pulse period, 0 = auto, min = 5 ms
    probe_pulse: int = -1, # the probe used for pulse sweep, e.g. gate probe for ID-VGS
):
    """Standard shared setup for FET IV measurements.
    """
    # reset instrument
    instr_b1500.write("*RST")
    instr_b1500.query("ERRX?") # clear any existing error message and ignore

    # enable channels: CN (pg 4-62)
    instr_b1500.write(f"CN {probe_sub},{probe_gate},{probe_drain},{probe_source}")
    query_error(instr_b1500)

    instr_b1500.write("FMT 1,1") # clear data buffer and set format (4-24, 4-25)
    instr_b1500.write("TSC 1")   # enables timestamp output
    instr_b1500.write("FL 0")    # set filter off
    query_error(instr_b1500)

    # instr_b1500.write("AV 10,1") # sets ADC number of samples for 1 data
    # query_error(instr_b1500)
    print("AAD")
    # select type of A/D converter (4-33, pg 353):
    #   ADD channel,type
    ADC_TYPE_HISPEED = 0
    ADC_TYPE_HIRES = 1
    ADC_TYPE_PULSE = 2
    adc_types = {
        probe_drain: ADC_TYPE_PULSE if pulsed and probe_pulse == probe_drain else ADC_TYPE_HISPEED,
        probe_source: ADC_TYPE_PULSE if pulsed and probe_pulse == probe_source else ADC_TYPE_HISPEED,
        probe_gate: ADC_TYPE_PULSE if pulsed and probe_pulse == probe_gate else ADC_TYPE_HISPEED,
        probe_sub: ADC_TYPE_HISPEED,
    }
    instr_b1500.write(f"AAD {probe_drain},{adc_types[probe_drain]}")
    query_error(instr_b1500)
    instr_b1500.write(f"AAD {probe_source},{adc_types[probe_source]}")
    query_error(instr_b1500)
    instr_b1500.write(f"AAD {probe_gate},{adc_types[probe_gate]}")
    query_error(instr_b1500)
    instr_b1500.write(f"AAD {probe_sub},{adc_types[probe_sub]}")
    query_error(instr_b1500)

    print("AIT")

    # Sets up high speed adc (4-38, pg 358)""
    #   AIT type,mode,N
    # type: ADC type of the A/D converter. Integer 0, 1, or 2
    #       0: high-speed ADC
    #       1: high-resolution ADC
    #       2: high-speed ADC for pulsed measurement
    # mode: ADC operation mode
    #       0: auto-mode, initial setting
    #       1: manual mode
    #       2: power line cycle (PLC) mode
    #       3: measurement time mode, not available for the high-res adc
    # N: coefficient used to define the integration time or the number of
    # averaging samples, integer expression, for mode = 0, 1 and 2. Or the
    # actual measurement time, numeric expression, for mode = 3... see table
    # 4-21 on page 4-39
    ADC_HISPEED_MODE_AUTO = 0
    ADC_HISPEED_MODE_MANUAL = 1
    ADC_HISPEED_MODE_POWER_LINE_CYCLE = 2
    ADC_HISPEED_MODE_MEASUREMENT_TIME = 3

    adc_hires_mode = ADC_HISPEED_MODE_AUTO
    adc_hires_sampling_coeff = 30
    instr_b1500.write(f"AIT {ADC_TYPE_HIRES},{adc_hires_mode},{adc_hires_sampling_coeff}")
    query_error(instr_b1500)

    # TODO: set pulsed ADC mode?

    print("DV")

    # zero voltage to probes, DV (pg 4-78) cmd sets DC voltage on channels:
    #   DV {probe},{vrange},{v},{icompliance}
    instr_b1500.write(f"DV {probe_gate},0,0,{ig_compliance}")
    print(f"DV {probe_gate},0,0,{ig_compliance}")
    query_error(instr_b1500)
    instr_b1500.write(f"DV {probe_sub},0,0,{id_compliance}")
    print(f"DV {probe_sub},0,0,{id_compliance}")
    query_error(instr_b1500)
    instr_b1500.write(f"DV {probe_drain},0,0,{id_compliance}")
    print(f"DV {probe_drain},0,0,{id_compliance}")
    query_error(instr_b1500)
    instr_b1500.write(f"DV {probe_source},0,0,{id_compliance}")
    print(f"DV {probe_source},0,0,{id_compliance}")
    query_error(instr_b1500)

    if pulsed:
        # Set pulse hold time, pulse width, and pulse period for a pulse
        # source set by the multi-channel PI, PV, PWI or PWV command. This command also
        # sets the trigger output delay time. (4-168, pg 488)
        #
        # MCPT: hold time, pulse period, measurement timing, and number measurements (4-154, pg. 465)
        #       MCPT hold[,period[,Mdelay[,average]]]
        # Parameters:
        #   - hold: hold time (in seconds), 10 ms resolution
        #   - period: pulse period (in seconds), 5 ms to 10 s, 0.1 ms resolution.
        #   - Mdelay: Measurement timing (in seconds) from the beginning of the pulse
        #       period to the beginning of the measurement. Default = 0, value set auto
        #       so measurement completes when peak to base begins
        #   - average: Number of measurements for averaging, default = 1.
        #
        # 
        # MCPNT: delay time and pulse widths (4-142, pg. 462)
        #       MCPNT chnum,delay,width
        # Parameters:
        #   - chnum: smu channel
        #   - delay: delay time (in seconds) from the beginning of the pulse period to the
        #           beginning of the transition from base to peak.
        #   - width: pulse width (in seconds)
        # For our HRSMU, delay is 0, width is 500 us to 2s, with 100 us resolution.
        
        print("MCPT")
        pulse_hold = 0.010 # 10 ms
        instr_b1500.write(f"MCPT {pulse_hold},{pulse_period}")
        print("MCPNT")
        instr_b1500.write(f"MCPNT {probe_pulse},0,{pulse_width}")

    print("MM")
    # set measurement mode to multi-channel staircase sweep (MODE = 16) (4-151, pg 471):
    if pulsed:
        # for pulsed, can only use one channel for pulse bias
        # MM mode,ch0,ch1,ch2,...
        # mm_mode = 4
        mm_mode = 28
        # instr_b1500.write(f"MM {mm_mode},{probe_pulse}")
        instr_b1500.write(f"MM {mm_mode},{probe_drain},{probe_source},{probe_gate}")
    else:
        # MM mode,ch0,ch1,ch2,...
        mm_mode = 16
        instr_b1500.write(f"MM {mm_mode},{probe_drain},{probe_source},{probe_gate}")
    query_error(instr_b1500)

    # set probe current measurement mode (4-62, pg 382):
    #   CMM ch,mode
    CMM_MODE_COMPLIANCE = 0
    CMM_MODE_CURRENT = 1
    CMM_MODE_VOLTAGE = 2
    CMM_MODE_FORCE = 3
    CMM_MODE_SYNC = 4
    cmm_mode = CMM_MODE_CURRENT
    instr_b1500.write(f"CMM {probe_drain},{cmm_mode}")
    instr_b1500.write(f"CMM {probe_source},{cmm_mode}")
    instr_b1500.write(f"CMM {probe_gate},{cmm_mode}")
    query_error(instr_b1500)

    # set auto-ranging (4-183 pg 503, page 339 for modes)
    RANGE_MODE_AUTO = 0      # auto
    RANGE_MODE_10PA = 9      # 10 pA limited auto
    RANGE_MODE_100PA = 10    # 100 pA limited auto
    RANGE_MODE_1NA = 11      # 1 nA limited auto
    RANGE_MODE_10NA = 12     # 10 nA limited auto
    RANGE_MODE_100NA = 13    # 100 nA limited auto
    RANGE_MODE_1UA = 14      # 1 uA limited auto
    RANGE_MODE_1MA = 17      # 1 mA limited auto
    RANGE_MODE_100UA_FIXED = -17 # 100 uA range fixed
    range_mode = RANGE_MODE_AUTO if pulsed else RANGE_MODE_1NA
    instr_b1500.write(f"RI {probe_source},{range_mode}")
    instr_b1500.write(f"RI {probe_drain},{range_mode}")
    instr_b1500.write(f"RI {probe_gate},{range_mode}")
    query_error(instr_b1500)

    # set hold time, delay time, and step delay time for the staircase sweep or
    # multi-channel sweep measurement (pg 4-246, pg 566)
    #   WT hold,delay[,Sdelay[,Tdelay[,Mdelay]]]
    # - hold: Hold time (in seconds) that is the wait time after starting the sweep
    #       measurement and before starting the delay time for the first step.
    #       0 to 655.35, with 10 ms resolution. Numeric expression.
    # - delay: Delay time (in seconds) that is the wait time after starting to
    #       force a step output and before starting a step measurement.
    #       0 to 65.535, with 0.1 ms resolution. Numeric expression.
    # - Sdelay : Step delay time (in seconds) that is the wait time after starting a step
    #       measurement and before starting to force the next step output value.
    #       If this parameter is not set, Sdelay will be 0.
    #       If Sdelay is shorter than the measurement time, the B1500 waits until
    #       the measurement completes, then forces the next step output.
    #
    # | hold | force | delay | Sdelay      | force | delay | Sdelay      |
    #                        | measure |                   | measure |
    #        |<----- repeat -------------->|
    #
    if pulsed:
        # pulsed staircase example on 3-29 (pg. 209) sets all to 0
        wt_hold = 0
        wt_delay = 0
        wt_sdelay = 0
    else:
        # standard staircase, these were valued used in past in stanford/mit
        wt_hold = 0.010  # 10 ms
        wt_delay = 0.010 # 10 ms, seems OK. if this is not 0, then it seems like first measurement is much lower than the others
        wt_sdelay = 0    # step delay
    instr_b1500.write(f"WT {wt_hold},{wt_delay},{wt_sdelay}")
    query_error(instr_b1500)

    # enable/disable automatic abort function WM (4-234, pg 554)
    #   The WM command enables or disables the automatic abort function for the
    #   staircase sweep sources and the pulsed sweep source. The automatic abort
    #   function stops the measurement when one of the following conditions
    #   occurs:
    #   - compliance on the measurement channel
    #   - compliance on the non-measurement channel
    #   - overflow on the AD converter
    #   - oscillation on any channel
    #   This command also sets the post measurement condition for the sweep
    #   sources. After the measurement is normally cmopleted, the staircase sweep
    #   sources force the value specified by the 'post' parameter, and the pulsed
    #   sweep source forces the pulse base value
    #
    #   WM abort[,post]
    # Parameters:
    # - abort : Automatic abort function.
    #       1: Disables the function. Initial setting.
    #       2: Enables the function.
    # - post : Source output value after the measurement is normally completed.
    #       1: Start value. Initial setting.
    #       2: Stop value.
    #       If this parameter is not set, the sweep sources force the start value.
    #
    # Output Data : The B1500 returns the data measured before an abort
    # condition is detected. Dummy data 199.999E+99 will be returned for the
    # data after abort
    WM_ABORT_DISABLE = 1
    WM_ABORT_ENABLE = 2
    WM_POST_START_VALUE = 1
    WM_POST_STOP_VALUE = 2
    instr_b1500.write(f"WM {WM_ABORT_DISABLE},{WM_POST_STOP_VALUE}")
    query_error(instr_b1500)

    # timestamp reset
    instr_b1500.write(f"TSR")
    query_error(instr_b1500)


class ProgramKeysightIdVgs(MeasurementProgram):
    """Implement Id-Vgs sweep with constant Vds biases.
    The Id-Vgs measurement is a staircase sweep (pg 2-8).
    The Vds is stepped at a constant bias on each step.
    This should output standard FET IDVG characteristics

          log ID
            |
            |                ...---^^^  VDS = 1
            |               /        
            |              /...---^^^^  VDS = 0.05
            |             //
            |   -._      //
            |   -_ ^--__//
            |     ^-_   /
            |        \_/
            |_______________________ VGS
    """
    name = "keysight_id_vgs"

    def default_config_string() -> str:
        return """
            probe_gate = 1
            probe_source = 8
            probe_drain = 4
            probe_sub = 9
            v_gs = { start = -1.2, stop = 1.2, step = 0.1 }
            v_ds = [-0.05, -1.2]
            v_sub = 0.0
            negate_id = true
            sweep_direction = "fr"
        """

    def run(
        instr_b1500=None,
        monitor_channel: EventChannel = None,
        signal_cancel=None,
        sweep_metadata: dict = {},
        probe_gate=1,
        probe_source=8,
        probe_drain=4,
        probe_sub=9,
        v_gs={
            "start": -1.2,
            "stop": 1.2,
            "step": 0.1,
        },
        v_ds=[-0.050, -1.2],
        v_sub=0.0,
        negate_id=True,
        sweep_direction="fr",
        id_compliance=0.010, # 10 mA drain complience
        ig_compliance=0.001, # 1 mA gate complience
        stop_on_error=True,
        yield_during_measurement=True,
        smu_slots={}, # map SMU number => actual slot number
        pulsed=False, # use DC pulsed mode
        pulse_width=0.0005, # pulse width (dc pulsed mode)
        pulse_period=0.010, # pulse period (dc pulsed mode)
        **kwargs,
    ) -> dict:
        """Run the program."""
        logging.info(f"probe_gate = {probe_gate}")
        logging.info(f"probe_source = {probe_source}")
        logging.info(f"probe_drain = {probe_drain}")
        logging.info(f"probe_sub = {probe_sub}")
        logging.info(f"v_ds = {v_ds}")
        logging.info(f"v_gs = {v_gs}")
        logging.info(f"v_sub = {v_sub}")
        logging.info(f"negate_id = {negate_id}")
        logging.info(f"sweep_direction = {sweep_direction}")
        logging.info(f"id_compliance = {id_compliance}")
        logging.info(f"ig_compliance = {ig_compliance}")
        logging.info(f"pulsed = {pulsed}")
        if pulsed:
            logging.info(f"pulse_width = {pulse_width}")
            logging.info(f"pulse_period = {pulse_period}")
        logging.info(f"smu_slots = {smu_slots}")
        
        if instr_b1500 is None:
            raise ValueError("Invalid instrument b1500 is None")
        
        # map smu probes to instrument slots
        if len(smu_slots) > 0:
            probe_gate = map_smu_to_slot(smu_slots, probe_gate)
            probe_source = map_smu_to_slot(smu_slots, probe_source)
            probe_drain = map_smu_to_slot(smu_slots, probe_drain)
            probe_sub = map_smu_to_slot(smu_slots, probe_sub)
            logging.info("Mapped SMU to slot:")
            logging.info(f"- probe_gate -> {probe_gate}")
            logging.info(f"- probe_source -> {probe_source}")
            logging.info(f"- probe_drain -> {probe_drain}")
            logging.info(f"- probe_sub -> {probe_sub}")

        # error check
        def query_error(instr_b1500):
            res = instr_b1500.query("ERRX?")
            if res[0:2] != "+0":
                logging.error(f"{res}")
                if stop_on_error:
                    raise RuntimeError(res)
        
        # convert v_ds and v_gs into a list of values depending on variable object type
        v_gs_range = into_sweep_range(v_gs)
        v_ds_range = into_sweep_range(v_ds)

        # maps string of sweep directions like "frf" => list of [SweepType.FORWARD_REVERSE, SweepType.FORWARD]
        sweeps = SweepType.parse_string(sweep_direction)
        
        # prepare output data matrices
        num_bias = len(v_ds_range)
        num_sweeps = SweepType.count_total_num_sweeps(sweeps)
        num_points = len(v_gs_range)
        data_shape = (num_bias, num_sweeps, num_points)

        v_ds_out = np.full(data_shape, np.nan)
        v_gs_out = np.full(data_shape, np.nan)
        i_d_out = np.full(data_shape, np.nan)
        i_s_out = np.full(data_shape, np.nan)
        i_g_out = np.full(data_shape, np.nan)
        # timestamps
        time_i_d_out = np.full(data_shape, np.nan)
        time_i_s_out = np.full(data_shape, np.nan)
        time_i_g_out = np.full(data_shape, np.nan)

        # measurement compliance derived settings
        pow_compliance = 2 * abs(id_compliance * np.max(np.abs(v_ds_range))) # power compliance [W]
        
        # standard keysight initialization for IV measurements
        measurement_keysight_b1500_setup(
            instr_b1500=instr_b1500,
            query_error=query_error,
            probe_gate=probe_gate,
            probe_source=probe_source,
            probe_drain=probe_drain,
            probe_sub=probe_sub,
            id_compliance=id_compliance,
            ig_compliance=ig_compliance,
            pow_compliance=pow_compliance,
            pulsed=pulsed,
            pulse_width=pulse_width,
            pulse_period=pulse_period,
            probe_pulse=probe_gate,
        )

        # ===========================================================
        # PERFORM SWEEP FOR EACH VDS AND SWEEP DIRECTION
        # ===========================================================

        # sweep state
        pulsed_str = "(VGS DC Pulsed)" if pulsed else "" # just print indicator that this is pulsed
        t_run_avg = None  # avg program step time
        cancelled = False # flag for program cancelled before done

        for idx_bias, v_ds_val in enumerate(v_ds_range):
            # round to mV
            v_ds_val = round(v_ds_val, 3)

            for idx_dir, sweep_type in SweepType.iter_with_sweep_index(sweeps):
                print(f"==============================")
                print(f"Measuring step (Vds = {v_ds_val} V)... {pulsed_str}")
                print(f"------------------------------")
                
                # select regular held staircase or pulsed staircase
                if pulsed:
                    sweep_command = sweep_type.b1500_mcpwnx_sweep_commands
                else:
                    sweep_command = sweep_type.b1500_wv_sweep_command

                # write voltage staircase waveform
                wv_range_mode = 0 # AUTO
                cmds = sweep_command(
                    ch=probe_gate,
                    range=wv_range_mode,
                    start=v_gs_range[0],
                    stop=v_gs_range[-1],
                    steps=len(v_gs_range),
                    icomp=id_compliance,
                    pcomp=None, # can trigger false errors
                )

                if pulsed: # this is array, kind of messy
                    for cmd in cmds:
                        print(cmd)
                        instr_b1500.write(cmd)
                else:
                    instr_b1500.write(cmds)

                query_error(instr_b1500)
                
                # write drain bias
                instr_b1500.write(f"DV {probe_drain},0,{v_ds_val},{id_compliance}")
                query_error(instr_b1500)
                
                # write bulk bias
                instr_b1500.write(f"DV {probe_sub},0,{v_sub},{id_compliance}")
                query_error(instr_b1500)
                
                # execute and wait for data response
                instr_b1500.write("XE")

                # starting time for step
                t_start = time()

                # yield green thread during measurement to let other tasks run
                if yield_during_measurement and t_run_avg is not None and t_run_avg > 0:
                    t_sleep = 0.9 * t_run_avg
                    logging.info(f"[ProgramKeysightIdVgs] SLEEPING: gevent.sleep({t_sleep:.3f})")
                    gevent.sleep(t_sleep)
                
                # set timeout (milliseconds)
                instr_b1500.timeout = 60 * 1000
                _opc = instr_b1500.query("*OPC?")
                instr_b1500.timeout = 10 * 1000
                query_error(instr_b1500)

                # update avg measurement time for accurate gevent sleep
                t_finish = time()
                t_run = t_finish - t_start
                t_run_avg = max(0, exp_moving_avg_with_init(t_run_avg, t_run, alpha=0.2, init_alpha=0.9))

                # zero probes after measurement
                instr_b1500.write(f"DV {probe_gate},0,0,{ig_compliance}")
                instr_b1500.write(f"DV {probe_sub},0,0,{id_compliance}")
                instr_b1500.write(f"DV {probe_drain},0,0,{id_compliance}")
                instr_b1500.write(f"DV {probe_source},0,0,{id_compliance}")
                query_error(instr_b1500)
                
                # number of bytes in output data buffer
                nbytes = int(instr_b1500.query("NUB?"))
                print(f"nbytes={nbytes}")
                buf = instr_b1500.read()
                print(f"buf, len(buf):{len(buf)}")
                print(buf)

                # parse vals strings into numbers
                vals = buf.strip().split(",")
                vals = parse_keysight_str_values(vals)

                # values chunked for each measurement point:
                #   [ [vgs0, id0, ig0] , [vgs1, id1, ig1], ... ]
                val_chunks = [ x for x in iter_chunks(vals, 7) ]
                print(f"val_chunks: {val_chunks}")

                # split val chunks into forward/reverse sweep components:
                if sweep_type == SweepType.FORWARD or sweep_type == SweepType.REVERSE:
                    sweep_chunks = [val_chunks]
                elif sweep_type == SweepType.FORWARD_REVERSE or sweep_type == SweepType.REVERSE_FORWARD:
                    sweep_chunks = [val_chunks[0:num_points], val_chunks[num_points:]]
                
                val_table = [] # values to print out to console for display

                for s, sweep_vals in enumerate(sweep_chunks):
                    for i, vals_chunk in enumerate(sweep_vals):
                        val_table.append([v_ds_val, vals_chunk[6], vals_chunk[1], vals_chunk[3], vals_chunk[5]])
                        
                        v_ds_out[idx_bias, idx_dir + s, i] = v_ds_val
                        v_gs_out[idx_bias, idx_dir + s, i] = vals_chunk[6]
                        i_d_out[idx_bias, idx_dir + s, i] = vals_chunk[1]
                        i_s_out[idx_bias, idx_dir + s, i] = vals_chunk[3]
                        i_g_out[idx_bias, idx_dir + s, i] = vals_chunk[5]
                        # timestamps
                        time_i_d_out[idx_bias, idx_dir + s, i] = vals_chunk[0]
                        time_i_s_out[idx_bias, idx_dir + s, i] = vals_chunk[2]
                        time_i_g_out[idx_bias, idx_dir + s, i] = vals_chunk[4]
                
                print(tabulate(val_table, headers=["v_ds [V]", "v_gs [V]", "i_d [A]", "i_s [A]", "i_g [A]"]))

                print("------------------------------")
                print(f"Finished step (Vds = {v_ds_val} V")
                print("==============================")

            # after each bias completes, send partial update
            if monitor_channel is not None and idx_bias < num_bias-1: # don't publish last step
                def task_update_program_status():
                    """Update program status."""
                    data={
                        "v_ds": v_ds_out,
                        "v_gs": v_gs_out,
                        "i_d": np.abs(i_d_out), # for easier plotting
                        "i_s": np.abs(i_s_out),
                        "i_g": np.abs(i_g_out),
                        "time_i_d": time_i_d_out,
                        "time_i_s": time_i_s_out,
                        "time_i_g": time_i_g_out,
                    }
                    data_cleaned = dict_np_array_to_json_array(data) # converts np ndarrays to regular lists and replace nan

                    monitor_channel.publish({
                        "metadata": {
                            "program": ProgramKeysightIdVgs.name,
                            "config": sweep_metadata,
                            "step": idx_bias,
                            "step_total": num_bias,
                        },
                        "data": data_cleaned,
                    })
                    
                gevent.spawn(task_update_program_status)

            # break after bias finished if cancelled
            if signal_cancel is not None and signal_cancel.is_cancelled():
                logging.info(f"[ProgramKeysightIdVgs] CANCELLING PROGRAM")
                cancelled = True
                break
        
        # zero voltages: DZ (pg 4-79)
        # The DZ command stores the settings (V/I output values, V/I output ranges, V/I
        # compliance values, and so on) and sets channels to 0 voltage.
        instr_b1500.write(f"DZ")

        # post-process: negate id
        if negate_id:
            i_d_out = -i_d_out
            i_s_out = -i_s_out

        return MeasurementResult(
            cancelled=cancelled,
            save_data=True, # save partial data even if cancelled
            data={
                "v_ds": v_ds_out,
                "v_gs": v_gs_out,
                "i_d": i_d_out,
                "i_s": i_s_out,
                "i_g": i_g_out,
                "time_i_d": time_i_d_out,
                "time_i_s": time_i_s_out,
                "time_i_g": time_i_g_out,
            },
        )


class ProgramKeysightIdVds(MeasurementProgram):
    """Implement Id-Vds sweep with constant Vgs biases.
    The Id-Vds measurement is a staircase sweep (pg 2-8).
    The Vgs is stepped at a constant bias on each step.
    
        ID
        |                           Vgs=3
        |                          __..--
        |                  __..--^^
        |          __..--^^
        |         /                   Vgs=2  
        |        /           ___...---^^^
        |      _/___...---^^^
        |    /                         Vgs=1
        |  _/________..........----------  
        |/__________________________________ Vds
    
    """

    name = "keysight_id_vds"

    def default_config_string() -> str:
        return """
            probe_gate = 1
            probe_source = 8
            probe_drain = 4
            probe_sub = 9
            v_gs = { start = 0.0, stop = -1.2, step = 0.4 }
            v_ds = { start = 0.0, stop = -2.0, step = 0.1 }
            v_sub = 0.0
            negate_id = true
            sweep_direction = "fr"
        """
    
    def run(
        instr_b1500=None,
        monitor_channel: EventChannel = None,
        signal_cancel=None,
        sweep_metadata: dict = {},
        probe_gate=3,
        probe_source=5,
        probe_drain=4,
        probe_sub=6,
        v_gs={
            "start": 0.0,
            "stop": -1.2,
            "step": 0.4,
        },
        v_ds={
            "start": 0.0,
            "stop": -2.0,
            "step": 0.1,
        },
        v_sub=0.0,
        negate_id=True,
        sweep_direction="fr",
        id_compliance=0.010, # 10 mA drain complience
        ig_compliance=0.001, # 1 mA gate complience
        stop_on_error=False,
        yield_during_measurement=True,
        smu_slots={}, # map SMU number => actual slot number
        pulsed=False, # use DC pulsed mode
        pulse_width=0.0005, # pulse width (dc pulsed mode)
        pulse_period=0.010, # pulse period (dc pulsed mode)
        **kwargs,
    ) -> dict:

        logging.basicConfig(
            level=logging.INFO,  # or DEBUG
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )

        """Run the program."""

        logging.info(f"probe_gate = {probe_gate}")
        logging.info(f"probe_source = {probe_source}")
        logging.info(f"probe_drain = {probe_drain}")
        logging.info(f"probe_sub = {probe_sub}")
        logging.info(f"v_ds = {v_ds}")
        logging.info(f"v_gs = {v_gs}")
        logging.info(f"v_sub = {v_sub}")
        logging.info(f"negate_id = {negate_id}")
        logging.info(f"sweep_direction = {sweep_direction}")
        logging.info(f"id_compliance = {id_compliance}")
        logging.info(f"ig_compliance = {ig_compliance}")
        logging.info(f"pulsed = {pulsed}")
        if pulsed:
            logging.info(f"pulse_width = {pulse_width}")
            logging.info(f"pulse_period = {pulse_period}")
        logging.info(f"smu_slots = {smu_slots}")
        
        if instr_b1500 is None:
            raise ValueError("Invalid instrument b1500 is None")
        
        # map smu probes to instrument slots
        if len(smu_slots) > 0:
            probe_gate = map_smu_to_slot(smu_slots, probe_gate)
            probe_source = map_smu_to_slot(smu_slots, probe_source)
            probe_drain = map_smu_to_slot(smu_slots, probe_drain)
            probe_sub = map_smu_to_slot(smu_slots, probe_sub)
            logging.info("Mapped SMU to slot:")
            logging.info(f"- probe_gate -> {probe_gate}")
            logging.info(f"- probe_source -> {probe_source}")
            logging.info(f"- probe_drain -> {probe_drain}")
            logging.info(f"- probe_sub -> {probe_sub}")

        # error check
        def query_error(instr_b1500):
            res = instr_b1500.query("ERRX?")
            if res[0:2] != "+0":
                logging.error(f"{res}")
                if stop_on_error:
                    raise RuntimeError(res)
        
        # convert v_ds and v_gs into a list of values depending on variable object type
        v_gs_range = into_sweep_range(v_gs)
        v_ds_range = into_sweep_range(v_ds)

        # maps string of sweep directions like "frf" => list of [SweepType.FORWARD_REVERSE, SweepType.FORWARD]
        sweeps = SweepType.parse_string(sweep_direction)
        
        # prepare output data matrices
        num_bias = len(v_gs_range)
        num_sweeps = SweepType.count_total_num_sweeps(sweeps)
        num_points = len(v_ds_range)
        data_shape = (num_bias, num_sweeps, num_points)

        v_ds_out = np.full(data_shape, np.nan)
        v_gs_out = np.full(data_shape, np.nan)
        i_d_out = np.full(data_shape, np.nan)
        i_s_out = np.full(data_shape, np.nan)
        i_g_out = np.full(data_shape, np.nan)
        # timestamps
        time_i_d_out = np.full(data_shape, np.nan)
        time_i_s_out = np.full(data_shape, np.nan)
        time_i_g_out = np.full(data_shape, np.nan)

        # measurement compliance settings
        pow_compliance = 2 * abs(id_compliance * np.max(np.abs(v_ds_range))) # power compliance [W], added some margin

        # standard keysight initialization for IV measurements
        measurement_keysight_b1500_setup(
            instr_b1500=instr_b1500,
            query_error=query_error,
            probe_gate=probe_gate,
            probe_source=probe_source,
            probe_drain=probe_drain,
            probe_sub=probe_sub,
            id_compliance=id_compliance,
            ig_compliance=ig_compliance,
            pow_compliance=pow_compliance,
            pulsed=pulsed,
            pulse_width=pulse_width,
            pulse_period=pulse_period,
            probe_pulse=probe_drain,
        )
        
        # ===========================================================
        # PERFORM SWEEP FOR EACH VDS AND SWEEP DIRECTION
        # ===========================================================

        # sweep state
        pulsed_str = "(VDS DC Pulsed)" if pulsed else "" # just print indicator that this is pulsed
        t_run_avg = None  # avg program step time
        cancelled = False # flag for program cancelled before done

        for idx_bias, v_gs_val in enumerate(v_gs_range):
            # round to mV
            v_gs_val = round(v_gs_val, 3) # round to mV

            for idx_dir, sweep_type in SweepType.iter_with_sweep_index(sweeps):
                print(f"==============================")
                print(f"Measuring step {idx_bias+1}/{len(v_gs_range)} (Vgs = {v_gs_val} V)... {pulsed_str}")
                print(f"------------------------------")

                # select regular held staircase or pulsed staircase
                if pulsed:
                    sweep_command = sweep_type.b1500_pwv_sweep_command
                else:
                    sweep_command = sweep_type.b1500_wv_sweep_command
                
                # write voltage staircase waveform
                wv_range_mode = 0 # AUTO
                instr_b1500.write(sweep_command(
                    ch=probe_drain,
                    range=wv_range_mode,
                    start=v_ds_range[0],
                    stop=v_ds_range[-1],
                    steps=len(v_ds_range),
                    icomp=id_compliance,
                    pcomp=None, # can trigger false errors
                ))
                query_error(instr_b1500)
                
                # write gate bias
                instr_b1500.write(f"DV {probe_gate},0,{v_gs_val},{ig_compliance}")
                query_error(instr_b1500)
                
                # write bulk bias
                instr_b1500.write(f"DV {probe_sub},0,{v_sub},{id_compliance}")
                query_error(instr_b1500)
                
                # execute and wait for data response
                instr_b1500.write("XE")

                # starting time for step
                t_start = time()

                # yield green thread during measurement to let other tasks run
                if yield_during_measurement and t_run_avg is not None and t_run_avg > 0:
                    t_sleep = 0.9 * t_run_avg
                    logging.info(f"[ProgramKeysightIdVds] SLEEPING: gevent.sleep({t_sleep:.3f})")
                    gevent.sleep(t_sleep)
                
                # set timeout (milliseconds)
                instr_b1500.timeout = 60 * 1000
                _opc = instr_b1500.query("*OPC?")
                instr_b1500.timeout = 10 * 1000
                query_error(instr_b1500)

                # update avg measurement time for accurate gevent sleep
                t_finish = time()
                t_run = t_finish - t_start
                t_run_avg = max(0, exp_moving_avg_with_init(t_run_avg, t_run, alpha=0.2, init_alpha=0.9))

                # zero probes after measurement
                instr_b1500.write(f"DV {probe_gate},0,0,{ig_compliance}")
                instr_b1500.write(f"DV {probe_sub},0,0,{id_compliance}")
                instr_b1500.write(f"DV {probe_drain},0,0,{id_compliance}")
                instr_b1500.write(f"DV {probe_source},0,0,{id_compliance}")
                query_error(instr_b1500)
                
                # number of bytes in output data buffer
                nbytes = int(instr_b1500.query("NUB?"))
                print(f"nbytes={nbytes}")
                buf = instr_b1500.read()
                print(buf)

                # parse vals strings into numbers
                vals = buf.strip().split(",")
                vals = parse_keysight_str_values(vals)

                # values chunked for each measurement point:
                #   [ [vgs0, id0, ig0] , [vgs1, id1, ig1], ... ]
                val_chunks = [ x for x in iter_chunks(vals, 7) ]
                print(val_chunks)

                # split val chunks into forward/reverse sweep components:
                if sweep_type == SweepType.FORWARD or sweep_type == SweepType.REVERSE:
                    sweep_chunks = [val_chunks]
                elif sweep_type == SweepType.FORWARD_REVERSE or sweep_type == SweepType.REVERSE_FORWARD:
                    sweep_chunks = [val_chunks[0:num_points], val_chunks[num_points:]]
                
                val_table = [] # values to print out to console for display

                for s, sweep_vals in enumerate(sweep_chunks):
                    for i, vals_chunk in enumerate(sweep_vals):
                        val_table.append([v_gs_val, vals_chunk[6], vals_chunk[1], vals_chunk[3], vals_chunk[5]])
                        
                        v_ds_out[idx_bias, idx_dir + s, i] = vals_chunk[6]
                        v_gs_out[idx_bias, idx_dir + s, i] = v_gs_val
                        i_d_out[idx_bias, idx_dir + s, i] = vals_chunk[1]
                        i_s_out[idx_bias, idx_dir + s, i] = vals_chunk[3]
                        i_g_out[idx_bias, idx_dir + s, i] = vals_chunk[5]
                        # timestamps
                        time_i_d_out[idx_bias, idx_dir + s, i] = vals_chunk[0]
                        time_i_s_out[idx_bias, idx_dir + s, i] = vals_chunk[2]
                        time_i_g_out[idx_bias, idx_dir + s, i] = vals_chunk[4]
                
                print(tabulate(val_table, headers=["v_gs [V]", "v_ds [V]", "i_d [A]", "i_s [A]", "i_g [A]"]))

                print("------------------------------")
                print(f"Finished step (Vgs = {v_gs_val} V")
                print("==============================")
            
            # after each bias completes, send partial update
            if monitor_channel is not None and idx_bias < num_bias-1: # don't publish last step
                def task_update_program_status():
                    """Update program status."""
                    data={
                        "v_ds": v_ds_out,
                        "v_gs": v_gs_out,
                        "i_d": np.abs(i_d_out), # for easier plotting
                        "i_s": np.abs(i_s_out),
                        "i_g": np.abs(i_g_out),
                        "time_i_d": time_i_d_out,
                        "time_i_s": time_i_s_out,
                        "time_i_g": time_i_g_out,
                    }
                    data_cleaned = dict_np_array_to_json_array(data) # converts np ndarrays to regular lists and replace nan

                    monitor_channel.publish({
                        "metadata": {
                            "program": ProgramKeysightIdVds.name,
                            "config": sweep_metadata,
                            "step": idx_bias,
                            "step_total": num_bias,
                        },
                        "data": data_cleaned,
                    })
                    
                gevent.spawn(task_update_program_status)

            # break after bias finished if cancelled
            if signal_cancel is not None and signal_cancel.is_cancelled():
                logging.info(f"[ProgramKeysightIdVds] CANCELLING PROGRAM")
                cancelled = True
                break
        
        # zero voltages: DZ (pg 4-79)
        # The DZ command stores the settings (V/I output values, V/I output ranges, V/I
        # compliance values, and so on) and sets channels to 0 voltage.
        instr_b1500.write(f"DZ")

        # post-process: negate id
        if negate_id:
            i_d_out = -i_d_out
            i_s_out = -i_s_out

        return MeasurementResult(
            cancelled=cancelled,
            data={
                "v_ds": v_ds_out,
                "v_gs": v_gs_out,
                "i_d": i_d_out,
                "i_s": i_s_out,
                "i_g": i_g_out,
                "time_i_d": time_i_d_out,
                "time_i_s": time_i_s_out,
                "time_i_g": time_i_g_out,
            },
        )


class ProgramKeysightIdVgsPulsedDC(MeasurementProgram):
    """Implement Id-Vgs sweep but with DC pulsed sweep. This is just a wrapper
    around ProgramKeysightIdVgs that makes sure the program is called with 
    `pulsed=True`. 
    """
    name = "keysight_id_vgs_pulsed_dc"

    def default_config_string() -> str:
        return """
            probe_gate = 1
            probe_source = 8
            probe_drain = 4
            probe_sub = 9
            v_gs = { start = -1.2, stop = 1.2, step = 0.1 }
            v_ds = [-0.05, -1.2]
            v_sub = 0.0
            negate_id = true
            sweep_direction = "fr"
            pulse_width = 0.0010
            pulse_period = 0.010
        """

    def run(
        **kwargs,
    ) -> dict:
        return ProgramKeysightIdVgs.run(pulsed=True, **kwargs)


class ProgramKeysightIdVdsPulsedDC(MeasurementProgram):
    """Implement Id-Vds sweep but with DC pulsed sweep. This is just a wrapper
    around ProgramKeysightIdVds that makes sure the program is called with 
    `pulsed=True`. 
    """
    name = "keysight_id_vds_pulsed_dc"

    def default_config_string() -> str:
        return """
            probe_gate = 1
            probe_source = 8
            probe_drain = 4
            probe_sub = 9
            v_gs = { start = 0.0, stop = -1.2, step = 0.4 }
            v_ds = { start = 0.0, stop = -2.0, step = 0.1 }
            v_sub = 0.0
            negate_id = true
            sweep_direction = "fr"
            pulse_width = 0.0005
            pulse_period = 0.010
        """

    def run(
        **kwargs,
    ) -> dict:
        return ProgramKeysightIdVds.run(pulsed=True, **kwargs)


if __name__ == "__main__":
    """
    Tests running the programs as standalone command-line module.
    """
    import argparse
    import os
    import json
    from controller.util.io import export_hdf5, export_mat
    from controller.backend import ControllerSettings
    print("Debugging message...", flush=True)

    parser = argparse.ArgumentParser(description="Run FET IV measurement.")

    parser.add_argument(
        "program",
        metavar="program",
        type=str,
        help="Program name, either 'idvg' or 'idvd'"
    )

    args = parser.parse_args()

    if args.program == "idvg":
        program = ProgramKeysightIdVgs
    elif args.program == "idvd":
        program = ProgramKeysightIdVds
    elif args.program == "idvg_pulsed_dc":
        program = ProgramKeysightIdVgsPulsedDC
    elif args.program == "idvd_pulsed_dc":
        program = ProgramKeysightIdVdsPulsedDC
    else:
        print("INVALID PROGRAM NAME: supported are 'idvg', 'idvd'")
        parser.print_help()
        exit()

    rm = pyvisa.ResourceManager()
    print(rm.list_resources())

    if len(rm.list_resources()) == 0:
        print("NO INSTRUMENTS FOUND")
        exit()
    
    # TODO: configure this shit

    # try to load default settings from "./settings/config.json"
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

    def run_measurement():
        print(f"RUNNING PROGRAM: {program.name}")
        try:
            result = program.run(
                instr_b1500=instr_b1500,
                smu_slots=config.smu_slots,
            )
            # print(result)
        except Exception as err:
            print(f"Measurement FAILED: {err}")
            instr_b1500.write(f"DZ") # ensure channels are zero-d
            print(traceback.format_exc())
        
        path_result_mat = f"./scripts/data/{program.name}.mat"
        path_result_h5 = f"./scripts/data/{program.name}.h5"
        export_hdf5(path_result_h5, result.data)
        export_mat(path_result_mat, result.data)
    
    # task = gevent.spawn(run_measurement)
    # gevent.joinall([task])
    run_measurement()
    # done, turn off
    print("MEASUREMENT DONE, TURNING OFF SMUs WITH CL")
    instr_b1500.write("CL")