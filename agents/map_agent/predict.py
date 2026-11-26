"""
predict.py - map-quality prediction from the temperature change since the map (plain code, no AI).

The sample decorrelates as its temperature moves away from the map's, so the check is modelled as
check = c0 * exp(-(dT / T_c)^2), dT = |kalman at the check - kalman at the last map| (Kalman readings only);
c0 is the check of a fresh map (about 0.86, limited by camera noise). c0 and T_c are fitted by least
squares on the (dT, check_map) pairs in the lab database, linearised as ln(check) = ln(c0) - dT^2 / T_c^2. Until min_points pairs exist the prediction is None (= always
check). After max_bad predictions off by more than max_error, the predictor is not trusted for the run.
With a form from staleness.py (PREDICTION form not "gaussian"), the check is modelled with that form instead:
c0 * exp(-a * dT^beta [- b * dt]), beta fixed by the form, dt = time since the last map; c0, a (and b) fitted the same way.
"""

from __future__ import annotations

import numpy as np

from agents.map_agent import staleness
from agents.core.state import LabState
from agents.map_agent.watchdog import kalman_at
from config import PREDICTION as SETTINGS
from config import TARGET_C


class MapPredictor:
    """Predicts check_map from the Kalman temperature change since the last map."""

    def __init__(self, state: LabState, run_id: int, form: dict | None = None):
        """form: a staleness.py form {"name", "beta", "time"}, or None for the original gaussian code path."""
        self.state, self.run_id, self.form = state, run_id, form
        self.n_bad = 0
        self.trusted = True

    def pairs(self, before_s: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """(dT (n,), check (n,)) for every check_map before lab time before_s, each against the map before it."""
        rows = self.state.query("SELECT t_s, kind, correlation FROM maps WHERE run_id = ? AND t_s < ? ORDER BY t_s, rowid",
                                (self.run_id, before_s))
        dT, check, k_map = [], [], None
        for row in rows:
            if row["kind"] == "measure":
                k_map = kalman_at(self.state, self.run_id, row["t_s"])
            elif k_map is not None:
                dT.append(abs(kalman_at(self.state, self.run_id, row["t_s"]) - k_map))
                check.append(row["correlation"])
        return np.array(dT), np.array(check)

    def fit(self, before_s: float = np.inf) -> tuple[float, float] | None:
        """(c0, T_c in C) from the pairs before before_s, or None with fewer than min_points pairs."""
        dT, check = self.pairs(before_s)
        if len(dT) < SETTINGS["min_points"]:
            return None
        # ln(check) = ln(c0) - a * dT^2 with a = 1 / T_c^2; checks near 0 are clipped so the log stays finite.
        design = np.column_stack([np.ones_like(dT), -dT ** 2])          # (n x 2)
        (ln_c0, a), *_ = np.linalg.lstsq(design, np.log(np.clip(check, 1e-3, 1.0)), rcond=None)
        return float(np.exp(ln_c0)), (float(1 / np.sqrt(a)) if a > 0 else np.inf)

    def last_map_s(self, now: float) -> float | None:
        """Lab time of the last measure_map at or before now (None without a map)."""
        rows = self.state.query("SELECT MAX(t_s) AS t FROM maps WHERE run_id = ? AND kind = 'measure' AND t_s <= ?",
                                (self.run_id, now))
        return rows[0]["t"]

    def dT_now(self, now: float) -> float | None:
        """|kalman now - kalman at the last map| (C), None without a map."""
        rows = self.state.query("SELECT MAX(t_s) AS t FROM maps WHERE run_id = ? AND kind = 'measure' AND t_s <= ?",
                                (self.run_id, now))
        if rows[0]["t"] is None:
            return None
        return abs(kalman_at(self.state, self.run_id, now) - kalman_at(self.state, self.run_id, rows[0]["t"]))

    def predict(self, now: float) -> tuple[float, float] | None:
        """(predicted check, dT) at lab time now, or None (not enough pairs, no map, or not trusted)."""
        if not self.trusted:
            return None
        if self.form is not None:
            return self._predict_form(now)
        fit, dT = self.fit(now), self.dT_now(now)
        if fit is None or dT is None:
            return None
        c0, T_c = fit
        return float(c0 * np.exp(-(dT / T_c) ** 2)), dT

    def _predict_form(self, now: float) -> tuple[float, float] | None:
        """(predicted check, dT) with the staleness form, or None (fewer than min_points pairs, or no map)."""
        dT, dt, check = staleness.pairs(self.state, self.run_id, now)
        dT_now, t_map = self.dT_now(now), self.last_map_s(now)
        if len(dT) < SETTINGS["min_points"] or dT_now is None:
            return None
        params = staleness.fit(self.form, dT, dt, check)
        return float(staleness.predict(params, dT_now, now - t_map)), dT_now

    def test(self, check: float, predicted: float) -> float:
        """Compare a real check with the prediction; returns |error| and stops trusting after max_bad bad ones."""
        error = abs(check - predicted)
        if error > SETTINGS["max_error"]:
            self.n_bad += 1
            self.trusted = self.n_bad < SETTINGS["max_bad"]
        return error


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c0_true, T_c_true = 0.86, 0.05
    state = LabState(":memory:")
    run_id = state.start_run("predict_test", 0, {})
    # Made-up run: a map every 600 s at the target, then the box drifts 0.1 C per 600 s; a check every 60 s.
    for second in range(1, 2401):
        state.add_temperature(run_id, second, TARGET_C + 0.1 * ((second - 1) % 600) / 600, TARGET_C, TARGET_C, 500)
    for k in range(4):
        state.add_map(run_id, k * 600 + 1, "measure", None, 60.0)
        for c in range(1, 10):
            dT = 0.1 * 60 * c / 600
            check = c0_true * np.exp(-(dT / T_c_true) ** 2) + rng.normal(0, 0.01)
            state.add_map(run_id, k * 600 + 1 + 60 * c, "check", float(check), 5.0)

    predictor = MapPredictor(state, run_id)
    c0, T_c = predictor.fit()
    dT, check = predictor.pairs()
    print(f"{len(dT)} pairs, dT 0..{dT.max():.3f} C: fitted c0 {c0:.3f} (true {c0_true}), T_c {T_c:.4f} C (true {T_c_true})")
    assert abs(c0 - c0_true) < 0.02 and abs(T_c - T_c_true) < 0.005
    few = predictor.fit(before_s=60 * SETTINGS["min_points"] + 1)      # min_points - 1 checks
    print(f"with {SETTINGS['min_points'] - 1} pairs: fit {few}, predict {MapPredictor(state, run_id).predict(240)}")
    assert few is None and MapPredictor(state, run_id).predict(240) is None
    p, dT_p = predictor.predict(1800 + 1 + 120)
    print(f"predict at t=1921 s: {p:.3f} for dT {dT_p:.3f} C (model {c0_true * np.exp(-(dT_p / T_c_true) ** 2):.3f})")
    for error_check in (p, p - 0.2, p - 0.2):
        predictor.test(error_check, p)
    print(f"after 2 bad predictions: trusted {predictor.trusted}, predict {predictor.predict(1921)}")
    assert not predictor.trusted and predictor.predict(1921) is None
    print("predict: fit, too few points and distrust: ok")
