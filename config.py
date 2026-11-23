"""All tunable numbers. Change them here only."""

import os

from env import load_env

load_env()     # .env (API keys, lab addresses) into os.environ before the settings below read it

TARGET_C = 20.0                    # box target [source: thesis]

OPTICS = {
    "focus_n_background": 200,     # random patterns for the unfocused background level (metrics.focus_test)
    # The sample is exactly X0 at T_ref_C. A property of the sample, not of the box: set_target() never changes it.
    "T_ref_C": TARGET_C,           # C [estimate: the sample is first mapped at the thesis target]
    "T_step_C": 0.02,              # C, grid step of the temperature chain (lab.TemperatureSample); rho = 0.9876
                                   # per step, the interpolation changes the correlation by < 1e-4 [estimate]
}

N = 64

MAIN = {
    "N": N,                        # SLM squares (inputs)
    "M": 64,                       # camera points (outputs)
    "image_shape": (8, 8),         # to show the M camera points as an image
    "L_ratio": 8,                  # L = L_ratio * N patterns
    "seed": 0,                     # fake lab (true matrix, patterns, noise)
    "ggs_seed": 1,                 # GGS random starts
    "test_seed": 5,                # fresh test patterns
    "noise_level": 0.04,           # SNR about 25, as measured in the GGS paper
    "phase_error": 0.3,            # SLM phase error in radians; [source: thesis]
    "flicker": 0.05,               # laser brightness change per photo
    "bit_depth": 8,                # camera bit depth
    "full_scale": 10 * N,          # fixed exposure; mean intensity ~ N, so speckle almost never saturates
    "n_test": 100,                 # test patterns for the prediction correlation
    "focus_target": 27,            # camera point for the focus image
    "max_iter": 3000,              # GGS iteration limit
    "tol": 1e-6,                   # GGS stop tolerance
    "n_restarts": 3,               # GGS restart rounds for stuck columns
    "restart_threshold": 0.25,     # good columns fit to about 0.2 with these imperfections, stuck columns to about 0.33
    "out_dir": "results/main",     # where plots and data are saved
}

DRIFT_TEST = {
    **MAIN,               # same lab, camera and GGS settings as main.py
    "duration_min": 5.0,           # test length [source: thesis, 4 to 5 minute test]
    "dt_s": 15,                    # time between photos in seconds (21 time points)
    "T0": TARGET_C,                # degrees C; box start temperature and thermostat target
    "warmup_s": 300,               # s, thermostat runs this long before the test (with control only)
    "T_c": 1.6,                    # degrees C, correlation temperature of the sample [estimate, source: thesis drift curve]
    "y_seed": 42,                  # the fixed random matrix Y of the temperature model
    "box_seed": 3,                 # sensor noise of the box simulation
    "ref_seed": 4,                 # noise of the separate reference photo for the output correlation
    # Long test: one map at t = 0, then long_min minutes, in each room of PID_TUNING["holds"].
    "long_min": 30.0,              # min [estimate: longer than several PID swings of 5-6 min]
    "long_dt_s": 60,               # s, time between test photos in the long test [estimate]
    "long_warmup_s": 600,          # s, with control: at the target this long before t = 0 [source: PID_TUNING warmup_s]
    "long_margin": 0.1,            # tuned PID must end at least this much above no control in the warm room [estimate]
    "out_dir": "results/drift_test",
}

THERMOSTAT = {
    "P_max": 64.54,          # W, cooling power at full duty, as the box air feels it [source: real box test 2026-11-18]
    "tau_cooler": 42.88,     # s, the cooling u follows duty / duty_max with this first-order lag [source: real box test 2026-11-18]
    "duty_max": 1023,        # 10-bit PWM, one MOSFET, cooling only [source: thermostat document]
    "dt": 0.05,              # s, Euler step = one control loop, 20 loops per second [source: thermostat document; real box test 2026-11-18: 20.2 integral updates per s]
    "target": TARGET_C,      # C [source: thesis]
    "C": 5000.0,             # J/K, heat capacity of the box air node, kept fixed in the fit [estimate]
    "G": 5.94,               # W/K, heat leak from the room to the box air [source: real box test 2026-11-18]
    "C_w": 10550.0,          # J/K, heat capacity of the walls and contents [source: real box test 2026-11-18]
    "G_w": 8.224,            # W/K, heat exchange between the walls and contents and the box air [source: real box test 2026-11-18]
    "T_room": 20.8,          # C, room temperature [source: real box test 2026-11-18, room sensor next to the box]
    "P_internal": 0.468,     # W, laser, SLM, camera, electronics; includes any offset between box and room sensor [source: real box test 2026-11-18]
    "sensor_std": 0.0103,    # C, raw sensor noise around a linear trend, readings in 0.01 C steps [source: real box test 2026-11-18]
    "sensor_step_C": 0.01,   # C, every reading is rounded to this step, like the real BME280 [source: real box test 2026-11-18, readings in 0.01 C steps]
    "spike_prob": 0.0002,    # per reading [estimate: the real log had 0 spikes in 1853 readings]
    "spike_min": 0.5,        # C [source: thermostat document]
    "spike_max": 1.0,        # C [source: thermostat document]
    "P_dist_std": 0.54,      # W, std of the slow extra heat dP [estimate from real rest phase: fit RMS 0.0145 C with sensor noise 0.0103 C, analysis/fit_real_box.py]
    "P_dist_tau_s": 60.0,    # s, correlation time of dP [estimate: laser and electronics heat changes about 10 % over minutes]
    "Q_T": 1e-8,             # C^2, Kalman process noise of T; best of 1e-8..1e-3 in the self-test sweep [tuned on the simulated box, 5 seeds]
    "Q_d": 1e0,              # W^2, Kalman process noise of the extra heat d; best of 1e-5..1e1 in the self-test sweep [tuned on the simulated box, 5 seeds]
    "P0_d": 100.0,           # W^2, start uncertainty of d (about +-10 W) [estimate]
    "max_skips_in_row": 20,  # lockout guard: 20 readings = 1 s [estimate: real spikes are single readings]
    "gate_sigma": 4.0,       # skip a reading more than 4 * sqrt(P_pred + R) from the prediction [estimate]
    "P0": 1.0,               # Kalman start uncertainty [as specified for this simulation]
    # Real PID [source: real box test 2026-11-18, the box web page]. Assumed firmware details [estimate]: one PID
    # update per 50 ms loop on the filtered temperature, error e = T - target in C, sum_e += e each loop (no dt),
    # de = e - e_prev per loop, duty clipped to 0..duty_max, sum_e clipped to +-duty_max / Ki and not frozen while the
    # duty is saturated (see freeze_integral_when_saturated), sum_e reset at a target change. The 20 Hz integral is
    # confirmed by the real hold (analysis/fit_real_box.py: 20.2 updates per s).
    "Kp": 200.0,
    "Ki": 4.0,
    "Kd": 40.0,
    "kalman_model_factors": {},   # setting -> factor for the Kalman filter's box model; empty: it knows the fitted box exactly [estimate]
    "freeze_integral_when_saturated": False,  # sum_e keeps counting at duty 0 or duty_max, only clipped to +-duty_max / Ki [source: real box test 2026-11-18, 20 C hold: duty stays 0 until 20.15 C, the unwinding matches the clip of 1023]
    "limit_C": 0.03,         # C, +-0.03 C box stability limit [source: thesis]
    "swing_band_C": 0.4,     # C, the PID swing of the real box stays inside target +- this [estimate: the simulated hold at 20 C swings about +-0.3 C]
    # Operating point of the real PID hold, for the swing-period self-test [source: real box test 2026-11-18]:
    # T_w_start and u_start are the fitted model's wall temperature and cooler state at the hold start
    # (analysis/fit_real_box.py, after 16 min of full cooling).
    "real_hold": {"target": 16.5, "T_room": 21.0, "T_start": 16.0, "T_w_start": 19.09, "u_start": 1.0,
                  "duration_s": 1800, "period_min_s": 240, "period_max_s": 420},
}
# Kalman measurement noise = sensor noise squared (C^2), matching Sensor [source: real box test 2026-11-18].
THERMOSTAT["R"] = THERMOSTAT["sensor_std"] ** 2

LAB_API = {
    "measure_map_s": 60.0,         # lab time for the L photos of measure_map [estimate]
    "focus_s": 1.0,                # lab time for one focus photo [estimate]
}
# check_map takes n_test photos: same time per photo as measure_map.
LAB_API["check_map_s"] = LAB_API["measure_map_s"] * MAIN["n_test"] / (MAIN["L_ratio"] * MAIN["N"])

SCENARIO = {
    "duration_s": 7200,            # s, 2 hours [estimate]
    "room_warm_at_s": 1800,        # s, the room starts to be warmer [estimate]
    "room_warm_by_C": 3.0,         # C, room temperature rise [estimate]
    "room_warm_for_s": 1200,       # s, the rise is spread over this time [estimate: a room warms over many minutes]
    "cooler_fail_at_s": 3600,      # s, the cooler fails completely [estimate]
    "cooler_fail_for_s": 600,      # s, how long it stays off [estimate]
    "target_change_at_s": 5400,    # s, a user asks for a new target [estimate]
    "new_target_C": 20.5,          # C, the new target [estimate]
}

BASELINE = {
    "step_s": 60,                  # s, time between focus photos [estimate]
    "check_every_s": 300,          # s, "rules": check the map this often [estimate]
    "check_threshold": 0.8,        # "rules": measure a new map when check_map is below this [estimate]
    "warn_after_s": 120,           # s, "rules": warn when |kalman - target| > limit_C for longer than this [estimate]
}

STATE = {
    "db_path": "results/lab_state.sqlite",   # shared lab database (SQLite file, local only)
    "temperature_every_s": 1,                # s, one temperature row this often
}

AGENT_LOOP = {
    "stable_s": 120,               # s, the box is stable if |kalman - target| < THERMOSTAT swing_band_C for this long [estimate]
    "retry_after_s": 60,           # s, run the loop again this soon when the map is stale but the box not stable [estimate]
    "use_watchdog": True,          # watchdog.py alarms wake the agent (all three False = the basic Map agent)
    "use_safety": True,            # safety.py reviews every plan before act()
    "use_prediction": True,        # predict.py may skip check_map when the predicted check is clearly good
    # Stale relative to what a fresh map reaches now (off: stale = check < BASELINE check_threshold, as before).
    "use_fresh_level": True,
    "fresh_n": 3,                  # fresh level = median verify check of the last fresh_n new maps [estimate]
    "stale_share": 0.9,            # stale if check < stale_share * fresh level [estimate]
    "min_gain": 0.03,              # a new map must verify this much better than the check before it [estimate]
    "max_backoff_s": 1800,         # s, longest wait before the next re-measure after maps that did not help [estimate]
}

# Normal operation, measured in a simulated 30 min hold at 20 C (seed 0): duty never at full power, the raw reading
# never unchanged for more than 6 s, |kalman - target| at most 0.37 C [source: simulated hold, run to choose the watchdog thresholds].
WATCHDOG = {
    "off_target_s": 120,           # s, "box off target": |kalman - target| > swing_band_C this long [estimate: = BASELINE warn_after_s]
    "moved_C": 2 * THERMOSTAT["swing_band_C"],   # C, "temperature moved since map": beyond two normal swings [estimate]
    "moved_avg_s": 300,            # s, "moved": mean Kalman over this, about one PID swing period [source: simulated swing 5.1-5.7 min, real_box fit]
    "target_change_quiet_s": 600,  # s, no "moved" alarm this long after a target change [estimate: a 0.5 C step is inside the band after 88 s (simulated), plus one moved_avg_s, rounded up]
    "duty_full_share": 0.95,       # "cooler at full power": duty >= this share of duty_max [estimate]
    "full_power_s": 60,            # s, ... for this long, with the box above target + swing_band_C and still warming [estimate: never at full power in the normal 20 C hold; in a 3 C warmer room up to 92 s at full power, but never above the band and warming (simulated 40 min, seed 0)]
    "stuck_s": 60,                 # s, "sensor stuck": raw reading unchanged this long [estimate: normal hold max 6 s]
    "eta_drop_share": 0.7,         # "focus dropped": last eta below this share of the reference eta [estimate]
    "eta_ref_s": 300,              # s, reference eta = mean eta of the first eta_ref_s after the last map [estimate]
    "alarm_cooldown_s": 600,       # s, one alarm wakes the agent at most this often [estimate]
    "useful_within_s": 300,        # s, evaluation only: a wake this soon after a scenario event counts as useful [estimate]
}

SAFETY = {
    "min_map_gap_s": 240,          # s, no measure_map sooner than this after the last one [estimate]
    "max_map_time_per_hour_s": 600,   # s, lab time on measure_map in the last hour (10 maps) [estimate]
}

PREDICTION = {
    "min_points": 5,               # (dT, check) pairs needed before the fit is used [estimate]
    "margin": 0.05,                # skip check_map only if predicted > check_threshold + margin [estimate]
    "max_error": 0.05,             # |check - predicted| above this is a bad prediction [estimate]
    "max_bad": 2,                  # after this many bad predictions, always check for the rest of the run [estimate]
}

AI_BRAIN = {
    "provider": "fake",            # "claude" | "gemini" | "openai" | "fake" (no network) | "rules"
    "test_provider": "claude",     # real provider the agents.loop self-test also runs when its key is found
    "models": {
        "claude": "claude-sonnet-5-5",   # [source: Anthropic model list, 2026-11]
        "gemini": "gemini-3.8-flash",    # [source: ai.google.dev/gemini-api/docs/models, 2026-11: recommended for new projects]
        "openai": "gpt-6.2-sol",         # [source: developers.openai.com/api/docs/models, 2026-11: balance of intelligence and cost]
    },
    # Names of the environment variables that hold the keys (the keys themselves live only in .env).
    "key_env": {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"},
    "timeout_s": 10,               # s, one API call, then fall back to the rules [estimate]
    "max_tokens": 300,             # answer length limit; the JSON answer is ~50 tokens [estimate]
    "temperature": 0,              # same observations -> same plan [estimate]
    "max_wait_s": 300,             # s, longest wait the Map agent's AI may ask for (one check_every_s) [estimate]
    "max_actions": 2,              # most actions in one plan (per loop) [estimate]
    "max_measure_map": 1,          # most measure_map actions in one plan (per loop) [estimate]
    "max_wait_total_s": 300,       # s, all waits in one plan together [estimate]
}

COMPARE = {
    "seeds": list(range(10)),      # full run (python -m experiments.compare full): every seed of LabAPI
    "test_seeds": [0, 1],          # self-test (python -m experiments.compare)
    "operators": ["never", "rules", "basic", "full"],   # basic / full Map agent: rules brain, fresh level on
    "se_factor": 2.0,              # a difference is real only if |mean paired difference| > this many standard errors
    "out_dir": "results/compare",
}

REAL_BOX = {
    "url": os.environ.get("BOX_URL", ""),                # box JSON (GET only); set in .env, see .env.example
    "interval_s": 2,                                     # s, one box row this often
    "room_url": os.environ.get("ROOM_URL", ""),          # room JSON, all channels (GET only); set in .env, see .env.example
    "room_channel": "biglab",                            # the room channel of the box
    "room_interval_s": 60,                               # s, room read this often (new samples every ~5-10 min)
    "out_dir": "results/real_box",
    "duty_max": 1023,                                    # 10-bit PWM full scale [source: box web page]
    "timeout_s": 1.5,                                    # s, box request timeout (shorter than interval_s)
    "room_timeout_s": 20,                                # s, room request timeout (the response holds 7 days of history)
    "print_every_s": 30,                                 # s, one status line this often
    "ca_file": "/etc/ssl/cert.pem",                      # macOS CA bundle; the python.org Python has none of its own
    "user_agent": "real-box-logger",                     # the room server answers 403 to the default Python-urllib agent
}

FIT_REAL_BOX = {
    "csv": "results/real_box/test1_cooling_2026-11-18/test1_cooling_2026-11-18.csv",   # real box cooling test, logged every 2 s (never changed)
    "start_clock": "2026-11-18 13:09:33",          # use rows from here on (target 30 C, the rest phase starts)
    "phases": ["rest", "cooling", "hold", "warm-up"],   # in order, one per target value in the CSV
    "fit_phases": ["rest", "cooling", "warm-up"],  # the hold is left out of the fit (it is the check)
    # Start values: G (W/K), P_internal (W), P_max (W), tau_cooler (s), C_w (J/K), G_w (W/K) [estimate]
    "start": {"G": 5.0, "P_internal": 1.0, "P_max": 40.0, "tau_cooler": 30.0, "C_w": 20000.0, "G_w": 10.0},
    "start_k": 0.02,                               # 1/K, start value of the "weaker cooler when colder" term [estimate]
    "max_iter": 100,                               # Levenberg-Marquardt iteration limit
    "swing_smooth_s": 20,                          # s, moving average before finding the swing peaks
    "swing_edge_s": 10,                            # s, peaks this close to the start or end of a record are not counted
    "swing_band_C": 0.05,                          # C, hysteresis: a new run above/below target starts only beyond target +- this
    "hold_long_s": 3600,                           # s, a longer simulated hold for the settled swing
    "seeds": [0, 1, 2, 3, 4],                      # sensor and heat-disturbance seeds of the simulated hold
    "dist_seeds": list(range(20)),                 # seeds for the heat-disturbance estimate (rest phase, cooler off)
    # 20 C hold test [source: real box test 2026-11-18, 20 C hold]: target 20 C from 14:54:55, box 20.17 C at the
    # start (settled, cooler off before), room 21.1 C.
    "hold20_csv": "results/real_box/test2_hold_20C_2026-11-18/test2_hold_20C_2026-11-18.csv",
    "hold20_start_clock": "2026-11-18 14:54:55",
    "hold20_room_C": 21.1,
    # 20 C hold test with the new PID (Kp 800, Ki 0.5, Kd 40) [source: real box test 2026-11-18, test3: a cooler cable came
    # loose at about 15:51, no cooling after that]; use it only until hold20_new_pid_end_clock.
    "hold20_new_pid_csv": "results/real_box/test3_hold_20C_new_pid_2026-11-18/test3_hold_20C_new_pid_2026-11-18.csv",
    "hold20_new_pid_end_clock": "2026-11-18 15:51:00",
    "out_dir": "results/real_box/fit",
}

PID_TUNING = {
    # Grid of PID gains, same firmware rules as the real PID (THERMOSTAT: 20 Hz, no dt, clip, integral not frozen).
    # First search (results_first_search.csv): best at Kp 800, Ki 1, at the edge of its grid; second search
    # (results_second_search.csv, exact Kalman model, no sensor steps, no cooler term) ran away to Kp 25877 [estimate].
    "Kp": [300, 400, 500, 600, 800, 1000, 1300, 1600, 2000, 2500],
    "Ki": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "Kd": 40.0,                        # [source: first search, Kd 0 to 1000 gave the same results; the firmware D term uses the change per 50 ms loop, which is tiny]
    "compare": [(800.0, 0.5, 40.0), (200.0, 4.0, 40.0)],   # PIDs of the real tests [source: real box tests 2026-11-18]
    # "What if" PID for the agents.loop self-test only; THERMOSTAT keeps the gains the real box runs today.
    "refined": (3000.0, 0.36, 40.0),   # [source: results/real_box/pid_tuning/results.csv, row "refined"]
    # Hold cases, each duration_s after warmup_s at 20 C: (name, room C) [source: real box test 2026-11-18, 20 C hold; estimate: warm room 20.8 + 3]
    "holds": [("a", 21.1), ("b", 23.8)],
    "target": TARGET_C,                # C
    "duration_s": 3600,                # s, recorded time per case
    "warmup_s": 600,                   # s, hold cases: at the target before recording, so the box and the PID are settled [estimate]
    "score_from_s": 900,               # s, hold cases: RMS, share and max after the first 15 min
    # Case d, like the real test 3: box at T_start, walls at it, cooler off, then target 20 C [source: real box test 2026-11-18, 15:42]
    "d_T_start": 20.53,
    "d_room_C": 21.1,
    "seed": 0,                         # one seed for every run (same sensor noise and heat disturbance)
    "n_robust": 10,                    # best grid PIDs that get the robustness runs
    "robust_factors": [("P_max", 0.8), ("P_max", 1.2), ("tau_cooler", 0.8), ("tau_cooler", 1.2)],   # true box only [estimate]
    "kalman_model_factors": {"P_max": 1.1, "tau_cooler": 0.9},   # the Kalman filter's box model is 10 % off the fit [estimate]
    "duty_noise_weight": 0.001,        # C per %: score += weight * std of the duty step per loop (% of duty_max) [estimate]
    "Kp_range": (300.0, 3000.0),       # grid and Nelder-Mead limits [estimate: Kp 3000 moves the duty by 30 per 0.01 C sensor step]
    "Ki_range": (0.02, 2.0),           # [estimate]
    "nm_max_evals": 60,                # Nelder-Mead on (log Kp, log Ki), score = robust RMS + cooler term
    "nm_step": 0.3,                    # start simplex: +0.3 in log Kp and in log Ki (about x1.35) [estimate]
    # Real test 3 (FIT_REAL_BOX hold20_new_pid_csv), used from its target change until FIT_REAL_BOX hold20_new_pid_end_clock.
    "test3_from_clock": "2026-11-18 15:42:48",
    "out_dir": "results/real_box/pid_tuning",
}
