"""
thermostat.py - the temperature-controlled box and its thermostat.

One loop step (every dt seconds):
    BME280 sensor reading z -> Kalman filter -> PID -> PWM duty -> cooler -> box temperature

Box model, two lumped temperatures: the box air T (where the sensor is) and the walls and contents T_w,
a slower heat store that only exchanges heat with the air:
    C   * dT/dt   = G * (T_room - T) + P_internal - P_max * u + G_w * (T_w - T) + dP(t)
    C_w * dT_w/dt = G_w * (T - T_w)
u is the cooling the cooler really gives (0..1): it follows duty / duty_max with a first-order lag
of time constant tau_cooler. dP(t) is a slowly varying extra heat (Ornstein-Uhlenbeck process) in the
true box only; the Kalman filter's model (BoxModel.step) does not know it.

Sources of the numbers (see the comments in config.THERMOSTAT):
- Real box test 2026-11-18 (analysis/fit_real_box.py): G, P_internal, P_max, tau_cooler, C_w, G_w
  (fitted with C fixed), the sensor noise, and the real PID gains (Kp 200, Ki 4, Kd 40).
- Thermostat document / thesis: dt (20 loops per second, confirmed by the real test), 10-bit cooling-only
  PWM, target 20 C, spike size.
- Kalman Q_T and Q_d tuned on the simulated box; the spike gate, the lockout guard and the start
  uncertainty of d are estimates.
- Estimates: C, T_room, the heat disturbance dP, and how often spikes happen.
"""

from __future__ import annotations

import numpy as np

from config import PID_TUNING, PLOTS, TARGET_C
from config import THERMOSTAT as SETTINGS


class BoxModel:
    """The box air: heat leaks in from the room, the optics heat it, the walls and contents (T_w) exchange
    heat with it, and the cooler removes heat with a delay (u follows the duty)."""

    def __init__(self, s: dict, T_w: float, u: float = 0.0):
        """Box constants from the settings dict s; start temperature T_w (C) of the walls, start cooling u (0..1)."""
        self.s = s
        self.T_w = T_w
        self.u = u

    def a(self) -> float:
        """dT_next / dT of one step (the model is linear in T); used by the Kalman filter."""
        s = self.s
        return 1.0 - (s["G"] + s["G_w"]) / s["C"] * s["dt"]

    def step(self, T: float, duty: int) -> float:
        """One Euler step of length dt: new air temperature (C) for air temperature T and PWM duty (0..1023).

        Also moves the cooler state u and the wall temperature T_w on by dt.
        """
        s = self.s
        # The cooler feels the duty only after a delay (first-order lag).
        self.u += (1.0 - np.exp(-s["dt"] / s["tau_cooler"])) * (duty / s["duty_max"] - self.u)
        # Heat flow into the air in W: leak from the room + internal heat - cooler + heat from the walls.
        power = (s["G"] * (s["T_room"] - T) + s["P_internal"] - s["P_max"] * self.u
                 + s["G_w"] * (self.T_w - T))
        self.T_w += s["G_w"] * (T - self.T_w) / s["C_w"] * s["dt"]
        # The heat capacity C turns a power into a rate of temperature change.
        return T + power / s["C"] * s["dt"]


class Sensor:
    """BME280 temperature sensor: true temperature + Gaussian noise + rare spikes, rounded to sensor_step_C."""

    def __init__(self, s: dict, rng: np.random.Generator):
        """Store the noise settings from s and the random generator rng."""
        self.s = s
        self.rng = rng

    def read(self, T_true: float) -> float:
        """One reading (C) of the true temperature T_true."""
        s = self.s
        z = T_true + self.rng.normal(0.0, s["sensor_std"])
        # Occasionally a reading jumps by 0.5 to 1.0 C (glitch), up or down.
        if self.rng.random() < s["spike_prob"]:
            z += self.rng.choice([-1.0, 1.0]) * self.rng.uniform(s["spike_min"], s["spike_max"])
        # The real BME280 readings come in sensor_step_C steps.
        return round(z / s["sensor_step_C"]) * s["sensor_step_C"]


class KalmanFilter:
    """Kalman filter with two unknowns: the box temperature T and an unknown extra heat d (W).

    In simple words: the filter keeps a best guess of T and of d (a warmer room, a weaker
    cooler, more laser heat - anything the box model does not know), and how sure it is of each.
    Predict: the box model says where T goes in one step for the duty the PID sent last step,
    plus the extra heat d (T_pred = BoxModel.step(T, duty) + d / C * dt); d is expected to stay
    the same. Both guesses get a little less sure every step (Q_T for T, Q_d for d).
    Update: only T is measured (noise R). The gain says how far to move toward the reading; if
    the readings keep coming out warmer than predicted, part of that difference goes into d, so
    the model learns the missing heat and the estimate no longer lags behind.
    Spike rejection: a reading more than gate_sigma * sqrt(P_TT + R) away from the prediction
    is skipped (counted in n_skipped). Lockout guard: real spikes are single readings, so after
    max_skips_in_row skipped readings in a row the box itself must have changed; then the filter
    forgets how sure it was (P back to the start values) and takes the reading (counted in n_resets).

    The uncertainty P is the 2 x 2 covariance of (T, d), kept as P_TT, P_Td and P_dd.
    """

    def __init__(self, s: dict, box: BoxModel):
        """Settings s (Q_T, Q_d, R, P0, P0_d, gate_sigma, max_skips_in_row), box: model for the predict step."""
        self.s = s
        self.box = box
        # The model is linear in T: T_next = a * T + (terms without T); b turns heat into temperature.
        self.a = box.a()
        self.b = s["dt"] / s["C"]
        self.x = None
        self.d = 0.0
        self.n_skipped = 0
        self.n_resets = 0
        self.skips_in_row = 0
        self.last_skipped = False

    def _reset_P(self) -> None:
        """Start uncertainty: P0 for T, P0_d for d, no correlation."""
        self.P_TT, self.P_Td, self.P_dd = self.s["P0"], 0.0, self.s["P0_d"]

    def update(self, z: float, duty_prev: int) -> float:
        """Filter one reading z (C), with duty_prev the duty of the last step; returns the estimate x (C)."""
        s = self.s
        if self.x is None:
            # Start from the first reading.
            self.x = z
            self._reset_P()
            return self.x
        a, b = self.a, self.b
        # Predict with the box model plus the extra heat d, one dt ahead.
        x_pred = self.box.step(self.x, duty_prev) + b * self.d
        P_TT = a * a * self.P_TT + 2 * a * b * self.P_Td + b * b * self.P_dd + s["Q_T"]
        P_Td = a * self.P_Td + b * self.P_dd
        P_dd = self.P_dd + s["Q_d"]
        # Spike rejection: a reading far outside the expected spread is a glitch; keep the prediction.
        self.last_skipped = abs(z - x_pred) > s["gate_sigma"] * np.sqrt(P_TT + s["R"])
        if self.last_skipped:
            self.skips_in_row += 1
            if self.skips_in_row < s["max_skips_in_row"]:
                self.n_skipped += 1
                self.x, self.P_TT, self.P_Td, self.P_dd = x_pred, P_TT, P_Td, P_dd
                return self.x
            # Lockout guard: too many "spikes" in a row means the box changed; start over from this reading.
            self.n_resets += 1
            self.last_skipped = False
            self._reset_P()
            P_TT, P_Td, P_dd = self.P_TT, self.P_Td, self.P_dd
        self.skips_in_row = 0
        # Update: only T is measured; the gains for T and d move both guesses toward the reading.
        K_T, K_d = P_TT / (P_TT + s["R"]), P_Td / (P_TT + s["R"])
        innovation = z - x_pred
        self.x = x_pred + K_T * innovation
        self.d = self.d + K_d * innovation
        self.P_TT, self.P_Td, self.P_dd = (1 - K_T) * P_TT, (1 - K_T) * P_Td, P_dd - K_d * P_Td
        return self.x

    def steady_state_gain(self, n: int = 200000) -> float:
        """Gain K_T after n updates without spikes (the uncertainty no longer changes), for the printout."""
        a, b, s = self.a, self.b, self.s
        P_TT, P_Td, P_dd = s["P0"], 0.0, s["P0_d"]
        for _ in range(n):
            P_TT, P_Td, P_dd = (a * a * P_TT + 2 * a * b * P_Td + b * b * P_dd + s["Q_T"],
                                a * P_Td + b * P_dd, P_dd + s["Q_d"])
            K_T, K_d = P_TT / (P_TT + s["R"]), P_Td / (P_TT + s["R"])
            P_TT, P_Td, P_dd = (1 - K_T) * P_TT, (1 - K_T) * P_Td, P_dd - K_d * P_Td
        return K_T


class PID:
    """PID controller: more cooling when too warm (P), when warm for long (I), when warming fast (D)."""

    def __init__(self, Kp: float, Ki: float, Kd: float, duty_max: int, freeze_when_saturated: bool = False):
        """Gains Kp, Ki, Kd (per loop step, source: thermostat code) and the maximum duty.

        freeze_when_saturated=False (the real firmware): the error sum keeps counting while the duty sits at
        0 or duty_max, only clipped; True: conditional integration, the sum stops while saturated.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.duty_max = duty_max
        self.freeze_when_saturated = freeze_when_saturated
        self.duty_last = None
        self.reset()

    def reset(self) -> None:
        """Forget the error history (used when the target changes)."""
        self.sum_e = 0.0
        self.e_prev = None

    def update(self, e: float) -> int:
        """Duty (0..duty_max) for the error e = T_filtered - T_target (C, positive = too warm)."""
        # Anti-windup: the sum cannot grow beyond what the integral term could ever use.
        limit = self.duty_max / self.Ki
        # Conditional integration (optional): while the cooler is already full on and it is still too warm (or
        # already off and still too cold), more integral cannot help, so do not let it build up.
        saturated = (self.duty_last == self.duty_max and e > 0) or (self.duty_last == 0 and e < 0)
        if not (self.freeze_when_saturated and saturated):
            self.sum_e = float(np.clip(self.sum_e + e, -limit, limit))
        de = 0.0 if self.e_prev is None else e - self.e_prev
        self.e_prev = e
        u = self.Kp * e + self.Ki * self.sum_e + self.Kd * de
        # The cooler can only cool: duty is clipped to 0..duty_max.
        self.duty_last = int(np.clip(u, 0, self.duty_max))
        return self.duty_last


class Thermostat:
    """The true box with its sensor, Kalman filter and PID, advanced one control loop step (dt) at a time."""

    def __init__(self, s: dict, seed: int, T_start: float, target: float, control: bool = True,
                 use_kalman: bool = True, T_w_start: float | None = None, u_start: float = 0.0):
        """Settings s, seed for the sensor noise (and, separately, the heat disturbance dP), start state.

        control=False: duty stays 0. use_kalman=False: the PID gets the raw reading.
        T_w_start: start wall temperature (default T_start, a settled box); u_start: start cooling (0..1).
        """
        self.s = s
        # The true box gets its own copy of the settings, so a change to the real box (e.g. a
        # warmer room or a failed cooler) does not reach the Kalman filter's model.
        T_w = T_start if T_w_start is None else T_w_start
        self.box = BoxModel(dict(s), T_w, u_start)
        self.sensor = Sensor(s, np.random.default_rng(seed))
        # The Kalman filter's box model may differ from the true box (kalman_model_factors scale its settings).
        s_kalman = {**s, **{key: s[key] * f for key, f in s["kalman_model_factors"].items()}}
        self.kalman = KalmanFilter(s_kalman, BoxModel(s_kalman, T_w, u_start))
        self.pid = PID(s["Kp"], s["Ki"], s["Kd"], s["duty_max"], s["freeze_integral_when_saturated"])
        self.T, self.target, self.control, self.use_kalman = T_start, target, control, use_kalman
        self.duty = 0
        self.z = self.x = None
        # Slow extra heat dP (Ornstein-Uhlenbeck): each step keeps a fraction a of the old value and
        # adds fresh noise, so dP stays at std P_dist_std and forgets itself over P_dist_tau_s.
        # Its own generator, so the sensor readings stay the same.
        self.rng_dist = np.random.default_rng([seed, 1])
        self.a = np.exp(-s["dt"] / s["P_dist_tau_s"])
        self.dP = self.rng_dist.normal(0.0, s["P_dist_std"])

    def step(self) -> None:
        """One loop: read the sensor (z), filter (x), set the duty, then move the true temperature T by dt."""
        s = self.s
        self.z = self.sensor.read(self.T)
        self.x = self.kalman.update(self.z, self.duty)      # duty is still the one sent last step
        measured = self.x if self.use_kalman else self.z
        self.duty = self.pid.update(measured - self.target) if self.control else 0
        self.T = self.box.step(self.T, self.duty) + self.dP / s["C"] * s["dt"]
        self.dP = self.a * self.dP + np.sqrt(1 - self.a * self.a) * self.rng_dist.normal(0.0, s["P_dist_std"])


def simulate(
    duration_s: float,
    control: bool = True,
    T_start: float = SETTINGS["target"],
    target: float = SETTINGS["target"],
    target_steps: list[tuple[float, float]] | None = None,
    settings: dict | None = None,
    seed: int = 0,
    warmup_s: float = 0.0,
    use_kalman: bool = True,
    box_changes: list[tuple[float, str, float]] | None = None,
    T_w_start: float | None = None,
    u_start: float = 0.0,
) -> dict[str, np.ndarray]:
    """Simulate the box and thermostat for duration_s seconds.

    control=False: the cooler stays off (duty 0). target_steps: optional list of
    (time_s, new_target); the PID integral is reset at every change.
    warmup_s: run this long first without recording; then time restarts at 0
    (the PID integral and the Kalman state are kept), like a box already settled.
    use_kalman=False: the PID gets the raw sensor reading instead of the Kalman estimate
    (the filter still runs, so "T_filtered" is still recorded).
    box_changes: optional list of (time_s, setting, value) changes to the true box only
    (e.g. (100, "P_max", 0.0) for a failed cooler); the Kalman model does not see them.
    T_w_start, u_start: start wall temperature (default T_start) and start cooling (0..1), see Thermostat.
    Returns: dict of arrays (n_steps,): "t_s", "T_true", "T_raw", "T_filtered",
             "duty", "target", "sum_e" (PID error sum, to check anti-windup) and
             "skipped" (1 where the Kalman filter rejected the reading as a spike) and
             "resets" (number of lockout-guard resets so far, counted from the start of the warm-up).
    """
    s = SETTINGS if settings is None else settings
    n = int(round(duration_s / s["dt"]))
    box = Thermostat(s, seed, T_start, target, control, use_kalman, T_w_start, u_start)
    steps = sorted(target_steps or [])
    changes = sorted(box_changes or [])

    out = {k: np.empty(n) for k in ["t_s", "T_true", "T_raw", "T_filtered", "duty", "target", "sum_e", "skipped",
                                    "resets"]}
    n_warm = int(round(warmup_s / s["dt"]))
    for k in range(n_warm + n):
        i = k - n_warm                    # recorded index; negative during the warm-up
        t = i * s["dt"]
        while steps and t >= steps[0][0]:
            box.target = steps.pop(0)[1]
            box.pid.reset()
        while changes and t >= changes[0][0]:
            _, key, value = changes.pop(0)
            box.box.s[key] = value
        T = box.T
        box.step()
        if i >= 0:
            out["t_s"][i], out["T_true"][i], out["T_raw"][i], out["T_filtered"][i] = t, T, box.z, box.x
            out["duty"][i], out["target"][i], out["sum_e"][i] = box.duty, box.target, box.pid.sum_e
            out["skipped"][i] = box.kalman.last_skipped
            out["resets"][i] = box.kalman.n_resets
    return out


def cooler_off_rise(s: dict, T_start: float, duration_s: float) -> float:
    """Exact rise (C) of the air temperature after duration_s with the cooler off, from a settled box at T_start.

    The two-node model without cooling is linear, x' = A x + c with x = (T, T_w), so
    x(t) = x_eq + V exp(L t) V^-1 (x0 - x_eq) with the eigenvalues L and eigenvectors V of A.
    """
    A = np.array([[-(s["G"] + s["G_w"]) / s["C"], s["G_w"] / s["C"]], [s["G_w"] / s["C_w"], -s["G_w"] / s["C_w"]]])
    c = np.array([(s["G"] * s["T_room"] + s["P_internal"]) / s["C"], 0.0])
    x_eq = -np.linalg.solve(A, c)
    L, V = np.linalg.eig(A)
    x = x_eq + V @ (np.exp(L * duration_s) * np.linalg.solve(V, np.array([T_start, T_start]) - x_eq))
    return float(x[0] - T_start)



def pid_gains(name: str) -> dict:
    """THERMOSTAT overrides for a PID by name: "real" (THERMOSTAT itself, empty overrides) or a key of PID_TUNING."""
    return {} if name == "real" else dict(zip(("Kp", "Ki", "Kd"), PID_TUNING[name]))


def pid_label(name: str) -> str:
    """"PID Kp/Ki (note)", e.g. "PID 3000/0.36 (tuned in simulation)"."""
    gains = {**SETTINGS, **pid_gains(name)}
    return f"PID {gains['Kp']:g}/{gains['Ki']:g} ({PLOTS['pid_notes'][name]})"


def pid_of(box_settings: dict | None) -> str:
    """The PID name ("real" or a PID_TUNING key) with these gains, or "" if none matches."""
    gains = {**SETTINGS, **(box_settings or {})}
    for name in ["real", *PID_TUNING]:
        if isinstance(PID_TUNING.get(name, ()), tuple) and all(gains[k] == v for k, v in pid_gains(name).items()) and \
                (name != "real" or all(gains[k] == SETTINGS[k] for k in ("Kp", "Ki", "Kd"))):
            return name
    return ""


def pid_text(box_settings: dict | None) -> str:
    """The label of these gains ("PID 3000/0.36 (tuned in simulation)"), or just "PID Kp/Ki" for other gains."""
    name = pid_of(box_settings)
    gains = {**SETTINGS, **(box_settings or {})}
    return pid_label(name) if name else f"PID {gains['Kp']:g}/{gains['Ki']:g}"


def pid_tag(box_settings: dict | None) -> str:
    """The PID part of a run name: "pid_<Kp>_<Ki>", e.g. "pid_3000_0.36"."""
    gains = {**SETTINGS, **(box_settings or {})}
    return f"pid_{gains['Kp']:g}_{gains['Ki']:g}"


def plot_pids() -> list[str]:
    """The PIDs the comparing plots show: the real box PID and the main PID (PLOTS main_pid), without repeats."""
    return list(dict.fromkeys(["real", PLOTS["main_pid"]]))

if __name__ == "__main__":
    import json
    from pathlib import Path

    from analysis import plots
    from analysis.fit_real_box import swing
    from config import FIT_REAL_BOX

    out = Path("results/thermostat")
    s = SETTINGS
    lim, band = s["limit_C"], s["swing_band_C"]
    swing_args = (FIT_REAL_BOX["swing_smooth_s"], FIT_REAL_BOX["swing_edge_s"], FIT_REAL_BOX["swing_band_C"])

    # 1. Hold at 20 C for 30 min (several swing periods).
    # After a 300 s warm-up the PID integral has already built up, as in a box that has been running.
    hold_s = 1800
    hold = simulate(hold_s, control=True, T_start=TARGET_C, target=TARGET_C, seed=0, warmup_s=300)
    T_hold = hold["T_true"]
    dev = np.abs(T_hold - TARGET_C).max()
    sw = swing(hold["t_s"], T_hold, TARGET_C, *swing_args)
    print(f"hold: min {T_hold.min():.4f}, max {T_hold.max():.4f}, std {T_hold.std():.4f} C, "
          f"max |T - {TARGET_C:g}| {dev:.4f} C (limit 0.5), mean duty {hold['duty'].mean():.0f}")
    print(f"hold at 20 C: swing period {sw['period_s'] / 60:.2f} min, mean max {sw['mean_max_C']:.3f} C, "
          f"mean min {sw['mean_min_C']:.3f} C (no limit: little cooling is needed at this operating point)")
    print(f"hold: share of time inside ±{lim:g} °C (thesis): {100 * np.mean(np.abs(T_hold - TARGET_C) <= lim):.0f}%, "
          f"inside ±{band:g} °C (swing band): {100 * np.mean(np.abs(T_hold - TARGET_C) <= band):.0f}%")
    print(f"Kalman: skipped {int(hold['skipped'].sum())} of {len(hold['skipped'])} readings as spikes")
    # Limit 0.5 C [estimate from the real box test 2026-11-18: swing 16.25..17.00 C around 16.5 C].
    assert dev < 0.5, "hold: deviation too large"

    # 1b. The swing period at the operating point of the real hold (16.5 C target, 21 C room, after full cooling).
    rh = s["real_hold"]
    real = simulate(rh["duration_s"], control=True, T_start=rh["T_start"], target=rh["target"],
                    settings={**s, "T_room": rh["T_room"]}, seed=0, T_w_start=rh["T_w_start"], u_start=rh["u_start"])
    sw_real = swing(real["t_s"], real["T_true"], rh["target"], *swing_args)
    print(f"hold at {rh['target']:g} C, room {rh['T_room']:g} C (real hold): swing period "
          f"{sw_real['period_s'] / 60:.2f} min (limit {rh['period_min_s'] / 60:g}..{rh['period_max_s'] / 60:g}; "
          f"real 5.5 min), mean max {sw_real['mean_max_C']:.3f} C, mean min {sw_real['mean_min_C']:.3f} C "
          f"(real 17.00 / 16.25 C), mean duty {real['duty'].mean():.0f}")
    assert rh["period_min_s"] <= sw_real["period_s"] <= rh["period_max_s"], "real hold: swing period outside the limit"

    # 3. Kalman filter during the hold.
    ratio = np.std(hold["T_raw"] - hold["T_true"]) / np.std(hold["T_filtered"] - hold["T_true"])
    print(f"Kalman noise reduction: std(raw - true) / std(filtered - true) = {ratio:.2f} (limit > 2.5)")
    assert ratio > 2.5, "Kalman filter reduces noise too little"
    K_ss = KalmanFilter(s, BoxModel(s, s["target"])).steady_state_gain()
    print(f"Kalman steady-state gain for T: K_ss = {K_ss:.4f} (about {1 / K_ss:.0f} readings averaged)")
    r_kalman = np.corrcoef(hold["T_filtered"], hold["T_true"])[0, 1]
    print(f"Kalman vs true (hold): correlation {r_kalman:.3f}, "
          f"error std {np.std(hold['T_filtered'] - hold['T_true']):.4f} C")

    # Choice of Q_T: the same hold for several Q_T, each over seeds 0..4. Best = smallest mean hold std;
    # among Q_T within 0.0002 C of that, the one with the smallest mean max |T - target|.
    sweep = {}
    for Q in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        runs = [simulate(300, control=True, T_start=TARGET_C, target=TARGET_C, seed=seed, warmup_s=300,
                         settings={**s, "Q_T": Q})["T_true"] for seed in range(5)]
        sweep[Q] = (np.mean([T.std() for T in runs]), np.mean([np.abs(T - TARGET_C).max() for T in runs]))
        print(f"Q_T sweep (5 seeds): Q_T {Q:.0e}: mean hold std {sweep[Q][0]:.4f} C, "
              f"mean max |T - {TARGET_C:g}| {sweep[Q][1]:.4f} C")
    best_std = min(v[0] for v in sweep.values())
    Q_best = min((Q for Q, v in sweep.items() if v[0] <= best_std + 0.0002), key=lambda Q: sweep[Q][1])
    print(f"Q_T sweep: best Q_T {Q_best:.0e} (SETTINGS uses {s['Q_T']:.0e})")

    # Choice of Q_d: the normal hold and a hold where the room warms by 3 C at once at 150 s (a stress
    # test, no +-limit_C requirement), seeds 0..4.
    # Best = smallest mean max |T - target| over both tests.
    warm = [(150.0, "T_room", s["T_room"] + 3.0)]
    sweep_d = {}
    for Q_d in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
        max_devs = [np.abs(simulate(300, control=True, T_start=TARGET_C, target=TARGET_C, seed=seed, warmup_s=300,
                                    settings={**s, "Q_d": Q_d}, box_changes=changes)["T_true"] - TARGET_C).max()
                    for changes in ([], warm) for seed in range(5)]
        sweep_d[Q_d] = (np.mean(max_devs[:5]), np.mean(max_devs[5:]), np.mean(max_devs))
        print(f"Q_d sweep (5 seeds): Q_d {Q_d:.0e}: mean max |T - {TARGET_C:g}| hold {sweep_d[Q_d][0]:.4f} C, "
              f"warm room {sweep_d[Q_d][1]:.4f} C, both {sweep_d[Q_d][2]:.4f} C")
    Q_d_best = min(sweep_d, key=lambda Q_d: sweep_d[Q_d][2])
    print(f"Q_d sweep: best Q_d {Q_d_best:.0e} (SETTINGS uses {s['Q_d']:.0e})")

    # 2. Step from target + 1 C to the target, 30 min, with the real PID, the refined PID (the limit is asserted for
    # it), and the PIDs the plots show (overrides only).
    names = list(dict.fromkeys(["real", "refined", PLOTS["main_pid"], "first_search"]))
    steps, settle, limit_text, limit_s = {}, {}, {}, {}
    for name in names:
        run = steps[name] = simulate(1800, control=True, T_start=TARGET_C + 1.0, target=TARGET_C, seed=1,
                                     settings={**s, **pid_gains(name)})
        outside = np.flatnonzero(np.abs(run["T_true"] - TARGET_C) >= band)
        settling = settle[name] = run["t_s"][outside[-1]] + s["dt"] if outside.size else 0.0
        print(f"step {TARGET_C + 1:g} -> {TARGET_C:g} C, {pid_label(name)}: inside ±{band:g} °C for good after "
              f"{settling / 60:.1f} min (limit 10), undershoot {TARGET_C - run['T_true'].min():.3f} C")
        # Limit 10 min [estimate: the real box cooled by 1 C in about 3 min at full power, then the swing takes over].
        if name == "refined":
            assert settling <= 600, "step: not inside the swing band after 10 min"
        elif name == "real" and settling > 600:
            print(f"  expected to fail with the real {pid_label(name)} (swings ±0.3 C)")
        # The +-limit_C time for the plot legend: inside the limit from then on until the end of the run.
        out_lim = np.flatnonzero(np.abs(run["T_true"] - TARGET_C) > s["limit_C"])
        end = run["t_s"][-1]
        if out_lim.size and run["t_s"][out_lim[-1]] >= end - 60:
            limit_text[name] = f"never inside ±{s['limit_C']:g} °C in {end / 60:.0f} min"
            limit_s[name] = None
        else:
            limit_s[name] = float(run["t_s"][out_lim[-1]] + s["dt"]) if out_lim.size else 0.0
            limit_text[name] = (f"stays inside ±{s['limit_C']:g} °C after "
                                f"{(run['t_s'][out_lim[-1]] + s['dt'] if out_lim.size else 0.0) / 60:.1f} min")
        print(f"  {pid_label(name)}: {limit_text[name]}")
    step = steps["real"]          # the real PID, for the Kalman check below
    w = step["t_s"] <= 60.0
    print(f"Kalman vs true (cooldown 0-60 s): correlation {np.corrcoef(step['T_filtered'][w], step['T_true'][w])[0, 1]:.4f}, "
          f"error std {np.std(step['T_filtered'][w] - step['T_true'][w]):.4f} C")

    # 4. Cooler off, compared with the exact solution of the fitted model (no disturbance, no Euler error).
    off = simulate(300, control=False, T_start=TARGET_C, seed=2)
    rise = off["T_true"][-1] - TARGET_C
    expected = cooler_off_rise(s, TARGET_C, 300)
    print(f"cooler off for 300 s: rise {rise:.3f} C, fitted model {expected:.3f} C (room {s['T_room']:g} C, "
          f"limit ±0.05 C)")
    # Limit ±0.05 C [estimate: the heat disturbance changes a 300 s rise by about 0.01 C].
    assert abs(rise - expected) < 0.05, "cooler off: rise differs from the fitted model"

    # 5. Anti-windup: an unreachable target must not let the error sum run away.
    windup = simulate(120, control=True, T_start=TARGET_C, target=5.0, seed=3)
    limit = s["duty_max"] / s["Ki"]
    print(f"anti-windup (target 5 C): max |sum_e| {np.abs(windup['sum_e']).max():.1f} (limit {limit:.1f})")
    assert np.abs(windup["sum_e"]).max() <= limit, "anti-windup: error sum exceeded its limit"

    # 6. The same hold without the Kalman filter: the PID sees the raw, noisy readings.
    hold_raw = simulate(1800, control=True, T_start=TARGET_C, target=TARGET_C, seed=0, warmup_s=300, use_kalman=False)
    dev_raw = np.abs(hold_raw["T_true"] - TARGET_C)
    print(f"hold without Kalman: std {hold_raw['T_true'].std():.4f} °C, "
          f"max |T - {TARGET_C:g}| {dev_raw.max():.4f} °C, share inside ±{lim:g} °C {100 * np.mean(dev_raw <= lim):.0f}%")

    # 7. The cooler fails (P_max = 0) for 60 s at 100 s of a 1200 s hold; the Kalman model does not know.
    fail = simulate(1200, control=True, T_start=TARGET_C, target=TARGET_C, seed=0, warmup_s=300,
                    box_changes=[(100.0, "P_max", 0.0), (160.0, "P_max", s["P_max"])])
    after = fail["t_s"] >= 160.0
    outside = fail["t_s"][after][np.abs(fail["T_true"][after] - TARGET_C) >= band]
    back = outside[-1] + s["dt"] - 160.0 if outside.size else 0.0
    print(f"cooler fails 100 to 160 s: max true temperature {fail['T_true'].max():.3f} C, "
          f"Kalman resets {int(fail['resets'][-1] - fail['resets'][0])}, "
          f"back within ±{band:g} C for good {back / 60:.1f} min after the cooler returns (limit 10)")
    # Limit 10 min [estimate: about two swing periods].
    assert back <= 600, "cooler failure: the box does not come back in time"

    # The plots: the main PID next to the real box PID, and the first-search PID as a third line.
    shown = list(dict.fromkeys(["real", PLOTS["main_pid"], "first_search"]))
    colors = dict(zip(shown, ["black", "darkorange", "tab:green"]))
    holds = {}
    for name in shown:
        gains = {**s, **pid_gains(name)}
        holds[name] = {kalman: simulate(1800, control=True, T_start=TARGET_C, target=TARGET_C, seed=0, warmup_s=300,
                                        settings=gains, use_kalman=kalman) for kalman in (True, False)}
    # Std of the duty change per control step (% of full duty), with and without Kalman, for plot 03 and the report.
    for name in ("refined", "first_search"):
        holds.setdefault(name, {kalman: simulate(1800, control=True, T_start=TARGET_C, target=TARGET_C, seed=0,
                                                 warmup_s=300, settings={**s, **pid_gains(name)}, use_kalman=kalman)
                                for kalman in (True, False)})
    duty_pct = {n: {k: 100.0 * h[k]["duty"] / s["duty_max"] for k in (True, False)} for n, h in holds.items()}
    duty_step = {n: {k: float(np.diff(d[k]).std()) for k in (True, False)} for n, d in duty_pct.items()}
    for n, d in duty_step.items():
        print(f"hold {TARGET_C:g} C, {pid_label(n)}: std of the duty change per control step {d[True]:.3f} % with Kalman, "
              f"{d[False]:.3f} % on the raw sensor ({d[False] / d[True]:.1f} times calmer with Kalman)")
    def window(name: str) -> tuple[float, float]:
        """The first window (from 1 min on) where the Kalman duty stays above duty_working_pct (the cooler works) and the
        raw sensor has no spike (no reading more than spike_min / 2 from the true temperature)."""
        t, kal = holds[name][True]["t_s"], duty_pct[name][True]
        raw_run = holds[name][False]
        spike = np.abs(raw_run["T_raw"] - raw_run["T_true"]) > s["spike_min"] / 2
        n = PLOTS["kalman_window_s"]
        for t0 in np.arange(60.0, t[-1] - n, 5.0):
            w = (t >= t0) & (t <= t0 + n)
            if kal[w].min() > PLOTS["duty_working_pct"] and not spike[w].any():
                return float(t0), float(t0 + n)
        return 60.0, 60.0 + n

    kalman_pids = ["real", "refined"]
    legend = {n: f"{pid_label(n)}: {limit_text[n]}" for n in shown}
    paths = [
        plots.plot_box_deviation(
            {legend[n]: (steps[n]["t_s"], steps[n]["T_true"], steps[n]["target"]) for n in shown},
            out / "01_pid_cooldown_21_to_20C.png",
            f"PID cooldown from {TARGET_C + 1:g} °C to {TARGET_C:g} °C, three PIDs (seed 1)",
            colors={legend[n]: colors[n] for n in shown}),
        plots.plot_kalman_error(hold["t_s"], hold["T_true"], hold["T_raw"], hold["T_filtered"],
                                out / "02_kalman_error_vs_raw_error.png",
                                f" ({pid_label('real')}, hold at {TARGET_C:g} °C, seed 0)"),
        plots.plot_kalman_summary(
            {pid_label(n): {label: (holds[n][k]["t_s"], holds[n][k]["T_true"], holds[n][k]["target"], duty_pct[n][k])
                            for k, label in [(True, "Kalman"), (False, "raw sensor")]} for n in kalman_pids},
            {pid_label(n): window(n) for n in kalman_pids},
            out / "03_pid_hold_20C_with_vs_without_kalman.png", "Kalman filter: same temperature, much calmer cooler (seed 0)",
            colors={"Kalman": "blue", "raw sensor": "red"}),
    ]
    # Numbers of this self-test for other tools (the Scientist): cooldown, hold, Kalman and duty per PID.
    raw_err, kal_err = hold["T_raw"] - hold["T_true"], hold["T_filtered"] - hold["T_true"]
    summary = {
        "limit_C": s["limit_C"], "swing_band_C": band, "target_C": TARGET_C,
        "kalman_error_std_C": {"raw sensor": float(raw_err.std()), "Kalman": float(kal_err.std()), "PID": pid_label("real"),
                               "hold_s": float(hold_s)},
        "pids": {pid_label(n): {
            "cooldown_inside_limit_s": limit_s.get(n), "cooldown_inside_band_s": float(settle[n]) if n in settle else None,
            "hold_std_C": {"Kalman": float(np.std(holds[n][True]["T_true"] - TARGET_C)),
                           "raw sensor": float(np.std(holds[n][False]["T_true"] - TARGET_C))},
            "hold_max_dev_C": {"Kalman": float(np.abs(holds[n][True]["T_true"] - TARGET_C).max()),
                               "raw sensor": float(np.abs(holds[n][False]["T_true"] - TARGET_C).max())},
            "duty_change_std_pct": {"Kalman": duty_step[n][True], "raw sensor": duty_step[n][False]},
        } for n in holds},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    paths.append(out / "summary.json")
    for p in paths:
        print(p)
    print("thermostat.py self-test passed")
