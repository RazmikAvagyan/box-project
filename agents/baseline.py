"""
baseline.py - a simple operator with fixed rules, to compare against the agents later.

run(policy) plays the 2-hour scenario (scenario.add_events) through LabAPI and takes a focus
photo every step_s. Two policies:
- "never": only focuses, with the one map measured at the start.
- "rules": every check_every_s it checks the map and measures a new one if check_map is below
  check_threshold; if the Kalman temperature stays more than swing_band_C from the target for longer
  than warn_after_s, it writes one warning line to the log (the rules cannot fix the box).
The policy only uses LabAPI; the true temperature is recorded for the plot and the summary only.
"""

from __future__ import annotations

import numpy as np

from agents import scenario
from agents.lab_api import LabAPI
from agents.state import LabState
from config import BASELINE as SETTINGS
from config import DRIFT_TEST, SCENARIO, STATE, THERMOSTAT

POLICIES = ("never", "rules")


def run(policy: str, seed: int = 0, state=None, box_settings: dict | None = None) -> dict:
    """Play the scenario with one policy; returns per-step arrays and the API (for its log).

    state: optional LabState; the run is recorded there under the policy name (actor = policy).
    box_settings: optional THERMOSTAT overrides for this run (see LabAPI).

    Returns: dict with arrays (n_steps,) "t_s", "eta", "kalman", "T_true", "target", "measured", "checked",
             plus "lab_time_s" (lab time spent in measure_map and check_map), "warnings" and "api".
    """
    s = SETTINGS
    api = LabAPI(seed=seed, state=state, run_name=policy, box_settings=box_settings)
    api.actor = policy
    scenario.add_events(api)
    api.measure_map()
    rec = {k: [] for k in ["t_s", "eta", "kalman", "T_true", "target", "measured", "checked"]}
    lab_time, warnings = 0.0, 0
    last_check, out_since, warned = api.clock, None, False

    for k in range(1, int(SCENARIO["duration_s"] // s["step_s"]) + 1):
        # Each step starts on the step_s grid (later only if the last step's work took longer).
        if api.clock < k * s["step_s"]:
            api.wait(k * s["step_s"] - api.clock)
        t_step = api.clock
        eta = api.focus(DRIFT_TEST["focus_target"])
        reading = api.get_temperature()
        measured = checked = False

        if policy == "rules":
            if api.clock - last_check >= s["check_every_s"]:
                t0 = api.clock
                checked = True
                if api.check_map() < s["check_threshold"]:
                    api.measure_map()
                    measured = True
                lab_time += api.clock - t0
                last_check = api.clock
            if abs(reading["kalman"] - reading["target"]) > THERMOSTAT["swing_band_C"]:
                out_since = t_step if out_since is None else out_since
                if not warned and t_step - out_since > s["warn_after_s"]:
                    api.note(f"box {reading['kalman'] - reading['target']:+.3f} C from target "
                             f"for {t_step - out_since:.0f} s; the rules cannot fix this", level="warning")
                    warnings += 1
                    warned = True
            else:
                out_since, warned = None, False

        for key, value in [("t_s", t_step), ("eta", eta), ("kalman", reading["kalman"]), ("T_true", api.box.T),
                           ("target", reading["target"]), ("measured", measured), ("checked", checked)]:
            rec[key].append(value)

    out = {key: np.array(values) for key, values in rec.items()}
    out.update({"lab_time_s": lab_time, "warnings": warnings, "api": api})
    return out


def scenario_events() -> list[tuple[float, str]]:
    """The scenario events as (time_s, label) for the numbered event lines of the plots."""
    fail_end = SCENARIO["cooler_fail_at_s"] + SCENARIO["cooler_fail_for_s"]
    return [(SCENARIO["room_warm_at_s"], f"room starts to warm (+{SCENARIO['room_warm_by_C']:g} °C over "
                                         f"{SCENARIO['room_warm_for_s'] / 60:g} min)"),
            (SCENARIO["room_warm_at_s"] + SCENARIO["room_warm_for_s"], "room warming ends"),
            (SCENARIO["cooler_fail_at_s"], "cooler fails"), (fail_end, "cooler back"),
            (SCENARIO["target_change_at_s"], f"target {SCENARIO['new_target_C']:g} °C")]


def summarize(r: dict) -> dict:
    """Summary of one run: eta statistics, map calls, lab time on them, warnings, badly timed new maps, and the
    share of steps with the true temperature within +-limit_C of the target (the thesis requirement)."""
    # A new map taken while the box is outside its swing band goes stale as soon as the box settles.
    useless = int(np.sum(r["measured"] & (np.abs(r["kalman"] - r["target"]) > THERMOSTAT["swing_band_C"])))
    return {
        "mean_eta": float(r["eta"].mean()),
        "min_eta": float(r["eta"].min()),
        "share_eta_above_30": float(np.mean(r["eta"] > 30)),
        "measure_map_calls": int(r["measured"].sum()),
        "check_map_calls": int(r["checked"].sum()),
        "lab_time_on_maps_s": float(r["lab_time_s"]),
        "warnings": int(r["warnings"]),
        "maps_while_box_off_target": useless,
        "share_within_limit": float(np.mean(np.abs(r["T_true"] - r["target"]) <= THERMOSTAT["limit_C"])),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from analysis import plots

    out = Path("results/baseline")
    out.mkdir(parents=True, exist_ok=True)
    state = LabState(STATE["db_path"])
    runs = {}
    for policy in POLICIES:
        state.delete_runs(policy)                    # a new run replaces the old one with the same name
        runs[policy] = run(policy, state=state)
    state.close()
    summary = {policy: summarize(r) for policy, r in runs.items()}

    for policy, sm in summary.items():
        print(f"{policy}: mean eta {sm['mean_eta']:.1f}, min eta {sm['min_eta']:.1f}, "
              f"eta > 30 in {100 * sm['share_eta_above_30']:.0f}% of steps, "
              f"measure_map {sm['measure_map_calls']}, check_map {sm['check_map_calls']}, "
              f"lab time on maps {sm['lab_time_on_maps_s']:.0f} s, warnings {sm['warnings']}, "
              f"new maps while the box was off target {sm['maps_while_box_off_target']}")
    r = runs["rules"]
    after = r["t_s"] >= SCENARIO["target_change_at_s"]
    n = runs["never"]
    print(f"after the target change: mean eta never {n['eta'][after].mean():.1f}, rules {r['eta'][after].mean():.1f}")
    print("rules log (maps and warnings):")
    for line in r["api"].get_log():
        if any(word in line for word in ("measure_map", "WARNING", "EVENT")):
            print(f"  {line}")

    header = "policy,t_s,eta,kalman,T_true,target,measured,checked"
    rows = [f"{p},{t:.0f},{e:.3f},{kal:.4f},{tr:.4f},{tg:.2f},{int(m)},{int(c)}"
            for p, rr in runs.items()
            for t, e, kal, tr, tg, m, c in zip(rr["t_s"], rr["eta"], rr["kalman"], rr["T_true"], rr["target"],
                                                rr["measured"], rr["checked"])]
    (out / "results.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    (out / "summary.json").write_text(json.dumps({"settings": SETTINGS, "results": summary}, indent=2))

    events = scenario_events()
    print(plots.plot_policy_comparison(
        {"never": (n["t_s"], n["eta"], n["measured"]), "rules": (r["t_s"], r["eta"], r["measured"])},
        events, out / "01_never_vs_rules_focus_brightness.png", colors={"never": "red", "rules": "blue"}))
    print(out / "results.csv")
    print(out / "summary.json")
