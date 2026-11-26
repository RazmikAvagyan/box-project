"""
staleness.py - the map-staleness formula chosen from the data (plain code, no AI).

A map check is modelled from dT (|Kalman temperature change| since the last map, C) and dt (time since the last map, s):
  exponential  c0 * exp(-2 * dT / T_c)          (the field correlation exp(-dT/T_c), squared: an intensity check)
  gaussian     c0 * exp(-(dT / T_c)^2)
  stretched    c0 * exp(-(dT / T_c)^beta), beta fitted in PREDICTION beta_range (1 = exponential, 2 = gaussian)
and each of them "+ time": times exp(-dt / tau) (time drift of the sample), only with at least time_min_points checks
whose dt spread over at least time_min_spread_s. All are fitted by least squares on ln(check):
ln(check) = ln(c0) - a * dT^beta - b * dt with a >= 0 and b >= 0 (a grid over beta for the stretched form), so a map
never looks better after the temperature or the time moved.

The choice: leave-one-out prediction error of every form (RMSE of the predicted check, each check predicted by a fit
without it), scored n * ln(MSE) + penalty_per_param * (number of parameters) (AIC-like); the lowest score wins. With
fewer than min_points checks there is no choice (the current form stays).
A form is a dict {"name", "beta", "time"} (beta None = fitted); the predictor uses the chosen one with its beta fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.core.state import LabState
from agents.map_agent.watchdog import kalman_at
from config import PREDICTION as SETTINGS

BASE = {"exponential": 1.0, "gaussian": 2.0, "stretched": None}         # name -> beta (None = fitted)
FORMS = {**{name: {"name": name, "beta": beta, "time": False} for name, beta in BASE.items()},
         **{f"{name} + time": {"name": f"{name} + time", "beta": beta, "time": True} for name, beta in BASE.items()}}
GAUSSIAN = FORMS["gaussian"]


def pairs(state: LabState, run_id: int, before_s: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dT (n,), dt (n,), check (n,)) for every check_map before lab time before_s, each against the map before it."""
    rows = state.query("SELECT t_s, kind, correlation FROM maps WHERE run_id = ? AND t_s < ? ORDER BY t_s, rowid",
                       (run_id, before_s))
    dT, dt, check, k_map, t_map = [], [], [], None, None
    for row in rows:
        if row["kind"] == "measure":
            k_map, t_map = kalman_at(state, run_id, row["t_s"]), row["t_s"]
        elif k_map is not None:
            dT.append(abs(kalman_at(state, run_id, row["t_s"]) - k_map))
            dt.append(row["t_s"] - t_map)
            check.append(row["correlation"])
    return np.array(dT), np.array(dt, dtype=float), np.array(check)


def _lstsq(dT: np.ndarray, dt: np.ndarray, y: np.ndarray, beta: float, time: bool) -> tuple[np.ndarray, float]:
    """Coefficients (ln c0, a[, b]) of ln(check) = ln c0 - a * dT^beta [- b * dt] with a, b >= 0 (ln c0 free), and the
    residual sum of squares. Bounded least squares by trying every set of decay terms held at 0 (at most 4 small
    solves): the best fit whose free decay terms are all >= 0 is the exact bounded optimum."""
    cols = [np.ones_like(dT), -dT ** beta] + ([-dt] if time else [])
    design = np.column_stack(cols)                                         # (n x 2 or 3)
    decay = list(range(1, design.shape[1]))
    best_coef, best_sse = None, np.inf
    for mask in range(2 ** len(decay)):
        free = [0] + [j for k, j in enumerate(decay) if not mask >> k & 1]
        sub, *_ = np.linalg.lstsq(design[:, free], y, rcond=None)
        if np.any(sub[1:] < 0):
            continue
        coef = np.zeros(design.shape[1])
        coef[free] = sub
        resid = y - design @ coef
        if resid @ resid < best_sse:
            best_coef, best_sse = coef, float(resid @ resid)
    return best_coef, best_sse


def fit(form: dict, dT: np.ndarray, dt: np.ndarray, check: np.ndarray) -> dict:
    """Parameters {"c0", "a", "b", "beta"} of one form on the pairs (b = 0 without the time term)."""
    y = np.log(np.clip(check, 1e-3, 1.0))
    if form["beta"] is None:
        lo, hi = SETTINGS["beta_range"]
        betas = np.arange(lo, hi + 1e-9, SETTINGS["beta_step"])
        beta = min(betas, key=lambda b: _lstsq(dT, dt, y, b, form["time"])[1])
    else:
        beta = form["beta"]
    coef, _ = _lstsq(dT, dt, y, beta, form["time"])
    return {"c0": float(np.exp(coef[0])), "a": float(coef[1]), "b": float(coef[2]) if form["time"] else 0.0,
            "beta": float(beta)}


def predict(params: dict, dT, dt) -> np.ndarray:
    """Predicted check c0 * exp(-a * dT^beta - b * dt)."""
    return params["c0"] * np.exp(-params["a"] * np.asarray(dT) ** params["beta"] - params["b"] * np.asarray(dt))


def readable(form: dict, params: dict) -> dict:
    """T_c in C (by the form's own definition), beta, tau in s of the time term, and c0."""
    a, beta = params["a"], params["beta"]
    if a <= 0:
        T_c = np.inf
    elif form["beta"] == 1.0:
        T_c = 2 / a
    else:
        T_c = a ** (-1 / beta)
    return {"c0": params["c0"], "T_c_C": float(T_c), "beta": beta,
            "tau_s": (float(1 / params["b"]) if params["b"] > 0 else np.inf) if form["time"] else None}


def n_params(form: dict) -> int:
    """c0 and T_c, plus beta if fitted, plus tau with the time term."""
    return 2 + (form["beta"] is None) + form["time"]


def time_allowed(dt: np.ndarray) -> bool:
    """Enough checks spread in time for the time term."""
    return len(dt) >= SETTINGS["time_min_points"] and float(np.ptp(dt)) >= SETTINGS["time_min_spread_s"]


def choose(dT: np.ndarray, dt: np.ndarray, check: np.ndarray) -> dict | None:
    """Every candidate form with its leave-one-out RMSE, score and parameters, and the chosen form (lowest score);
    None with fewer than min_points checks."""
    n = len(check)
    if n < SETTINGS["min_points"]:
        return None
    forms = {}
    for name, form in FORMS.items():
        if form["time"] and not time_allowed(dt):
            continue
        errors = []
        for i in range(n):
            keep = np.arange(n) != i
            params = fit(form, dT[keep], dt[keep], check[keep])
            errors.append(float(predict(params, dT[i], dt[i]) - check[i]))
        mse = float(np.mean(np.square(errors)))
        params = fit(form, dT, dt, check)
        forms[name] = {"loo_rmse": float(np.sqrt(mse)), "k": n_params(form),
                       "score": n * float(np.log(max(mse, 1e-12))) + SETTINGS["penalty_per_param"] * n_params(form),
                       "params": params, "readable": readable(form, params)}
    chosen = min(forms, key=lambda name: forms[name]["score"])
    form = {**FORMS[chosen], "beta": forms[chosen]["params"]["beta"]}       # beta fixed for the predictor
    return {"n": n, "forms": forms, "chosen": chosen, "form": form}


def setup_key(pid_text: str, time_drift: str) -> str:
    """The setup a choice belongs to, e.g. "PID 200/4 (real box today), time drift none"."""
    return f"{pid_text}, time drift {time_drift}"


def save_choice(setup: str, form: dict, n: int, loo_rmse: dict, run_name: str,
                path: str | Path = SETTINGS["forms_file"]) -> None:
    """Remember the chosen form of a setup (the latest analysis wins) in the forms file, with the number of checks
    and every form's leave-one-out RMSE."""
    path = Path(path)
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[setup] = {"form": form, "run": run_name, "checks": n, "loo_rmse": loo_rmse}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2))


def resolve(setting: str, setup: str | None, path: str | Path = SETTINGS["forms_file"]) -> dict | None:
    """The predictor's form for PREDICTION form = setting: None for "gaussian" (the original code path), the named
    form, or with "auto" the latest chosen form of this setup from the forms file (None = gaussian if there is none)."""
    if setting == "gaussian":
        return None
    if setting != "auto":
        return FORMS[setting]
    path = Path(path)
    saved = json.loads(path.read_text()) if path.exists() else {}
    entry = saved.get(setup) if setup else None
    return None if entry is None else entry["form"]


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    # Made-up checks from known forms: dT up to 0.6 C, dt up to 600 s, noise 0.01.
    dT, dt = rng.uniform(0, 0.6, 40), rng.uniform(30, 600, 40)
    truths = {"exponential": {"c0": 0.86, "a": 2 / 1.6, "b": 0.0, "beta": 1.0},
              "gaussian": {"c0": 0.86, "a": 1 / 0.5 ** 2, "b": 0.0, "beta": 2.0},
              "exponential + time": {"c0": 0.86, "a": 2 / 1.6, "b": 1 / 900, "beta": 1.0}}
    for name, true in truths.items():
        check = predict(true, dT, dt) + rng.normal(0, 0.01, len(dT))
        choice = choose(dT, dt, check)
        print(f"made-up {name:20} -> chosen {choice['chosen']:20} (beta {choice['form']['beta']:.2f}); LOO RMSE "
              + ", ".join(f"{n} {f['loo_rmse']:.4f}" for n, f in choice["forms"].items()))
        base = name.split(" + ")[0]
        ok = choice["chosen"].split(" + ")[0] in (base, "stretched") and ("time" in choice["chosen"]) == ("time" in name)
        if choice["chosen"].startswith("stretched"):
            ok &= abs(choice["form"]["beta"] - true["beta"]) <= 0.3
        assert ok, (name, choice["chosen"], choice["form"])
    # Noisy data where the temperature barely matters: an unbounded fit gives a < 0 (a check rising with dT); the
    # bounded fit holds a at 0, so the prediction is flat, never rising.
    flat_rng = np.random.default_rng(3)
    dT_flat, dt_flat = flat_rng.uniform(0, 0.05, 15), flat_rng.uniform(30, 600, 15)
    check_flat = 0.85 + 0.6 * dT_flat + flat_rng.normal(0, 0.01, 15)          # rises with dT: noise-like trend
    y_flat = np.log(check_flat)
    free_fit, *_ = np.linalg.lstsq(np.column_stack([np.ones(15), -dT_flat]), y_flat, rcond=None)
    params = fit(FORMS["exponential"], dT_flat, dt_flat, check_flat)
    rising = predict(params, np.array([0.0, 0.05]), np.zeros(2))
    print(f"unbounded fit a {free_fit[1]:.3f} < 0; bounded fit a {params['a']:.3f}, predicted {rising[0]:.4f} at dT 0 "
          f"and {rising[1]:.4f} at dT 0.05")
    assert free_fit[1] < 0 and params["a"] == 0.0 and rising[0] == rising[1]
    for name, form in FORMS.items():
        p = fit(form, dT_flat, dt_flat, check_flat)
        assert p["a"] >= 0 and p["b"] >= 0, (name, p)
    assert choose(dT[:SETTINGS["min_points"] - 1], dt[:SETTINGS["min_points"] - 1], check[:SETTINGS["min_points"] - 1]) is None
    assert not time_allowed(dt[:SETTINGS["time_min_points"] - 1])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "forms.json"
        assert resolve("auto", "setup A", path) is None and resolve("gaussian", "setup A", path) is None
        save_choice("setup A", choice["form"], choice["n"], {}, "test run", path)
        assert resolve("auto", "setup A", path) == choice["form"] and resolve("auto", "setup B", path) is None
        assert resolve("exponential", None, path) == FORMS["exponential"]
    print("made-up forms, too few checks, the forms file: ok")

    # Simulated runs (full Map agent, seed 0, original gaussian predictor): without time drift the choice is exponential
    # (or stretched with beta near 1); with time drift 30 min it has the time term.
    from agents.map_agent import loop
    from box_simulation.thermostat import pid_gains, pid_label
    for pid in ("real", "refined"):
        for tau in (None, 30):
            state = LabState(":memory:")
            r = loop.run(state=state, brain="rules", switches={"use_analyst": False}, box_settings=pid_gains(pid) or None,
                         lab_settings={"time_drift_tau_min": tau}, run_name="staleness_test")
            choice = choose(*pairs(state, r["api"].run_id))
            print(f"{pid_label(pid)}, time drift {tau or 'none'}: {choice['n']} checks -> {choice['chosen']} "
                  f"(beta {choice['form']['beta']:.2f}); LOO RMSE " +
                  ", ".join(f"{n} {f['loo_rmse']:.4f}" for n, f in choice["forms"].items()))
            if tau is None:
                assert choice["chosen"] == "exponential" or (choice["chosen"] == "stretched"
                                                             and abs(choice["form"]["beta"] - 1) <= 0.2)
            else:
                assert choice["form"]["time"]
    print("staleness.py self-test passed")
