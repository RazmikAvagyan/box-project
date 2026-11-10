"""All tunable numbers. Change them here only."""

TARGET_C = 20.0                    # box target [source: thesis]

OPTICS = {
    "focus_n_background": 200,     # random patterns for the unfocused background level (metrics.focus_test)
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
    "out_dir": "results/drift_test",
}

THERMOSTAT = {
    "P_max": 300.0,          # W, cooling power at full duty [S-2-12 sheet: 250 to 300 W]
    "duty_max": 1023,        # 10-bit PWM, one MOSFET, cooling only [source: thermostat document]
    "dt": 0.05,              # s, Euler step = one control loop, 20 loops per second [source: thermostat document]
    "target": TARGET_C,      # C [source: thesis]
    "C": 5000.0,             # J/K, heat capacity of the box [estimate]
    "G": 8.0,                # W/K, heat leak to the room [estimate]
    "T_room": 22.0,          # C, room temperature [estimate]
    "P_internal": 30.0,      # W, laser, SLM, camera, electronics [estimate]
    "sensor_std": 0.1,       # C, raw noise about +-0.3 C taken as ~3 standard deviations [source: thermostat document]
    "spike_prob": 0.002,     # per reading [estimate]
    "spike_min": 0.5,        # C [source: thermostat document]
    "spike_max": 1.0,        # C [source: thermostat document]
    "P_dist_std": 3.0,       # W, std of the slow extra heat dP [estimate: laser and electronics heat changes about 10 % over minutes]
    "P_dist_tau_s": 60.0,    # s, correlation time of dP [estimate: laser and electronics heat changes about 10 % over minutes]
    "Q_T": 1e-8,             # C^2, Kalman process noise of T; best of 1e-8..1e-3 in the self-test sweep [tuned on the simulated box, 5 seeds]
    "Q_d": 1e0,              # W^2, Kalman process noise of the extra heat d; best of 1e-5..1e1 in the self-test sweep [tuned on the simulated box, 5 seeds]
    "P0_d": 100.0,           # W^2, start uncertainty of d (about +-10 W) [estimate]
    "max_skips_in_row": 20,  # lockout guard: 20 readings = 1 s [estimate: real spikes are single readings]
    "gate_sigma": 4.0,       # skip a reading more than 4 * sqrt(P_pred + R) from the prediction [estimate]
    "P0": 1.0,               # Kalman start uncertainty [as specified for this simulation]
    "Kp": 3000.0,            # PID gains, hand-tuned [source: thermostat document]
    "Ki": 0.8,               # [source: thermostat document]
    "Kd": 50.0,              # [source: thermostat document]
    "limit_C": 0.03,         # C, +-0.03 C box stability limit [source: thesis]
}
# Kalman measurement noise = sensor noise squared (C^2), matching Sensor [source: sensor settings].
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
