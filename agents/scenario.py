"""
scenario.py - "world" events in the lab that the operator cannot see directly.

add_events(api) puts timed events into api.events; LabAPI runs each one when the lab clock
reaches its time. The events change the true box directly (never through the LabAPI
functions), so the Kalman filter's model does not know about them:
a room that warms gradually, a cooler that fails for a while, and a user asking for a new target.
"""

from __future__ import annotations

from config import SCENARIO as SETTINGS


def add_events(api) -> None:
    """Add the SETTINGS events to api.events, a list of (time_s, function)."""
    s = SETTINGS
    true_box = api.box.box.s                     # settings of the true box only (not the Kalman model)
    P_max = true_box["P_max"]

    # The room warms gradually: a small step every 60 s, room_warm_by_C in total after room_warm_for_s.
    n_steps = int(s["room_warm_for_s"] // 60)
    step_C = s["room_warm_by_C"] / n_steps

    def room_starts_warming():
        api._log(f"EVENT room starts to warm, +{s['room_warm_by_C']:g} C over {s['room_warm_for_s'] / 60:g} min")

    def room_warms_a_bit():
        true_box["T_room"] += step_C

    def room_warms_last_bit():
        true_box["T_room"] += step_C
        api._log(f"EVENT room is now {s['room_warm_by_C']:g} C warmer")

    def cooler_fails():
        true_box["P_max"] = 0.0
        api._log("EVENT cooler fails")

    def cooler_back():
        true_box["P_max"] = P_max
        api._log("EVENT cooler works again")

    def target_changes():
        api.box.target = s["new_target_C"]
        api._log(f"EVENT user asks for target {s['new_target_C']:g} C")

    api.events += [
        (s["room_warm_at_s"], room_starts_warming),
        *[(s["room_warm_at_s"] + 60 * k, room_warms_a_bit) for k in range(1, n_steps)],
        (s["room_warm_at_s"] + 60 * n_steps, room_warms_last_bit),
        (s["cooler_fail_at_s"], cooler_fails),
        (s["cooler_fail_at_s"] + s["cooler_fail_for_s"], cooler_back),
        (s["target_change_at_s"], target_changes),
    ]


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from agents.lab_api import LabAPI
    from analysis import plots
    from config import THERMOSTAT

    s = SETTINGS
    api = LabAPI(seed=0)
    add_events(api)

    # No operator: only time passes. Record every 10 s (the true temperature only for this test).
    t, T_true, T_kalman, target, T_room, duty, P_max = [], [], [], [], [], [], []
    for _ in range(int(s["duration_s"] // 10)):
        api.wait(10)
        t.append(api.clock)
        T_true.append(api.box.T)
        T_kalman.append(api.box.x)
        target.append(api.box.target)
        T_room.append(api.box.box.s["T_room"])
        duty.append(api.box.duty)
        P_max.append(api.box.box.s["P_max"])
    t, T_true, T_kalman, target, T_room, duty, P_max = map(np.array, (t, T_true, T_kalman, target, T_room, duty, P_max))
    dev = np.abs(T_true - target)
    lim = THERMOSTAT["limit_C"]

    def max_dev(t_from: float, t_to: float) -> float:
        """Max |true - target| (C) for t_from <= t < t_to."""
        return dev[(t >= t_from) & (t < t_to)].max()

    def settle_time(t_from: float, t_to: float) -> str:
        """Time after t_from until |true - target| stays within the limit up to t_to, as text."""
        w = (t >= t_from) & (t < t_to)
        outside = t[w][dev[w] > lim]
        if outside.size == 0:
            return "0 s (never left)"
        if outside[-1] >= t[w][-1]:
            return "not within the limit before the next event"
        return f"{outside[-1] + 10 - t_from:.0f} s"

    fail_end = s["cooler_fail_at_s"] + s["cooler_fail_for_s"]
    print(f"max |true - target| before {s['room_warm_at_s'] / 60:g} min: {max_dev(0, s['room_warm_at_s']):.4f} C "
          f"(limit {lim:g})")
    print(f"max |true - target| with the warm room, {s['room_warm_at_s'] / 60:g} to {s['cooler_fail_at_s'] / 60:g} min: "
          f"{max_dev(s['room_warm_at_s'], s['cooler_fail_at_s']):.4f} C (limit {lim:g})")
    w = (t >= s["cooler_fail_at_s"]) & (t < fail_end)
    print(f"cooler failure: max true temperature {T_true[w].max():.3f} C "
          f"({T_true[w].max() - target[w][0]:+.3f} C from target)")
    print(f"after the cooler works again: within ±{lim:g} C after {settle_time(fail_end, s['target_change_at_s'])}")
    print(f"after the target change to {s['new_target_C']:g} C: within ±{lim:g} C after "
          f"{settle_time(s['target_change_at_s'], s['duration_s'] + 10)}")
    print("events:")
    for line in api.get_log():
        if "EVENT" in line:
            print(f"  {line}")

    events = [(s["room_warm_at_s"], f"room starts to warm (+{s['room_warm_by_C']:g} °C over "
                                    f"{s['room_warm_for_s'] / 60:g} min)"),
              (s["room_warm_at_s"] + s["room_warm_for_s"], "room warming ends"),
              (s["cooler_fail_at_s"], "cooler fails"), (fail_end, "cooler back"),
              (s["target_change_at_s"], f"target {s['new_target_C']:g} °C")]
    out = Path("results/scenario")
    print(plots.plot_scenario_overview(t, T_true, T_kalman, target, T_room, events,
                                       (s["cooler_fail_at_s"], fail_end), out / "01_box_and_room_temperature_overview.png"))
    print(plots.plot_scenario(t, T_true, T_kalman, target, events, out / "02_box_temperature_zoom_near_target.png",
                              T_room=T_room))
    # The duty the PID asks for, and the cooling the box really gets (0 while the cooler is broken).
    asked = duty / THERMOSTAT["duty_max"] * 100
    real = asked * P_max / THERMOSTAT["P_max"]
    print(plots.plot_cooler_power(t, asked, real, events, (s["cooler_fail_at_s"], fail_end),
                                  out / "03_cooler_power_during_scenario.png"))
