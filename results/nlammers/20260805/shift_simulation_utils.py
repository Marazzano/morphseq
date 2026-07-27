"""Simulate expected effective developmental stage under temperature-shift
protocols, using the EMPIRICAL temperature-dependent developmental rate observed
in the hotfish data (NOT the Kimmel 0.055T-0.57 formula).

Empirical rate derivation
-------------------------
Hotfish embryos sat at their treatment temperature continuously from ~6 hpf to
collection (timepoints 24/30/36 h).  Modeling stage as
    stage(T, t) = s6 + rate_emp(T) * (t - 6)
with a SHARED stage-at-6h ``s6`` and a per-temperature ``rate_emp(T)`` fit jointly
by least squares gives clean empirical rates (cohort means of ``mdl_stage_hpf``).
The rates are non-monotonic in T: they rise to a peak near 32C then FALL at
33.5/35C (the hot-end developmental breakdown).  A degree-3 polynomial
``rate_emp(T)`` captures that shape (interpolates 34C, mild extrapolation to 20C).

"Effective stage" = accumulated developmental progress in 28C-equivalent hpf:
    eff_stage(t) = s6_effective + integral_0^t rate_emp(T(tau)) dtau
Because rate_emp is expressed in stage-hpf per real-hour already, the trajectory
is just piecewise-linear accumulation of rate over the temperature history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import morph_stage_reliability_utils as mr


# --------------------------------------------------------------------------- #
# Empirical rate curve
# --------------------------------------------------------------------------- #

BASELINE_TEMP = 28.0     # temperature of phases 1 and 3 (the shift-back target)
BASELINE_START_H = 6.0   # all embryos at 28C for the first 6 h


def fit_empirical_rate(joint: pd.DataFrame | None = None, *,
                       stage_col: str = "mdl_stage_hpf",
                       baseline_h: float = BASELINE_START_H,
                       degree: int = 3):
    """Fit the empirical developmental-rate-vs-temperature curve from the
    hotfish cohort means.  Returns dict with:
      s6         : shared stage at ``baseline_h`` (hpf)
      temps      : measured temperatures
      rate_meas  : per-temperature empirical rate (stage-hpf / real-hour)
      rate_poly  : np.poly1d, degree-``degree`` rate_emp(T)
      rate_fn    : callable T -> rate (uses the polynomial)
    """
    if joint is None:
        joint = pd.read_csv(mr.CACHE_DIR / "joint_141_morph_seq.csv")
    g = joint.groupby(["temperature", "timepoint"])[stage_col].mean().reset_index()
    temps = sorted(g["temperature"].unique())

    # joint LS: stage = s6 + rate_T * (timepoint - baseline_h)
    X, y = [], []
    for _, r in g.iterrows():
        row = np.zeros(1 + len(temps))
        row[0] = 1.0
        row[1 + temps.index(r["temperature"])] = r["timepoint"] - baseline_h
        X.append(row)
        y.append(r[stage_col])
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    s6 = float(coef[0])
    rate_meas = {float(T): float(rt) for T, rt in zip(temps, coef[1:])}

    rate_poly = np.poly1d(np.polyfit(temps, [rate_meas[T] for T in temps], degree))

    def rate_fn(T):
        return float(rate_poly(T))

    return {"s6": s6, "temps": [float(t) for t in temps],
            "rate_meas": rate_meas, "rate_poly": rate_poly, "rate_fn": rate_fn}


# --------------------------------------------------------------------------- #
# Three-phase temperature history + effective-stage integration
# --------------------------------------------------------------------------- #

def temperature_at(t, treatment_T, *, baseline_temp=BASELINE_TEMP,
                   phase1_end=BASELINE_START_H, shiftback_h=22.0):
    """Piecewise temperature history:
      [0, phase1_end)      -> baseline_temp (28C)
      [phase1_end, shiftback_h) -> treatment_T
      [shiftback_h, inf)   -> baseline_temp (28C)
    """
    t = np.asarray(t, dtype=float)
    T = np.full_like(t, baseline_temp)
    T[(t >= phase1_end) & (t < shiftback_h)] = treatment_T
    return T


def effective_stage_trajectory(treatment_T, rate_fn, *, s6, t_grid,
                               baseline_temp=BASELINE_TEMP,
                               phase1_end=BASELINE_START_H, shiftback_h=22.0):
    """Effective (28C-equivalent) stage vs real time for one treatment cohort.

    Integrates rate_emp(T(tau)) over the temperature history.  Anchored so that
    at ``phase1_end`` the stage equals ``s6`` (all cohorts share the 6h stage,
    since all are at baseline through phase 1).  Returns stage array aligned to
    ``t_grid``.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    Tvec = temperature_at(t_grid, treatment_T, baseline_temp=baseline_temp,
                          phase1_end=phase1_end, shiftback_h=shiftback_h)
    rates = np.array([rate_fn(T) for T in Tvec])
    # cumulative integral of rate over time via trapezoid, anchored at phase1_end
    dt = np.diff(t_grid, prepend=t_grid[0])
    cum = np.cumsum(rates * dt)
    # stage at phase1_end should be s6; find offset
    i6 = int(np.argmin(np.abs(t_grid - phase1_end)))
    stage = s6 + (cum - cum[i6])
    return stage


def simulate_cohorts(treatment_temps, rate_fit, *, t_grid=None,
                     shiftback_h=22.0, baseline_temp=BASELINE_TEMP,
                     phase1_end=BASELINE_START_H, t_max=40.0, n_grid=801):
    """Effective-stage trajectories for every treatment cohort.

    Returns (traj_df, meta) where traj_df is long-form
    [treatment_T, real_time_h, eff_stage] and meta[T] holds the effective stage
    at the shift-back moment (the 'stage varies at 22h' quantity) and at
    ``phase1_end``.
    """
    if t_grid is None:
        t_grid = np.linspace(0, t_max, n_grid)
    rate_fn, s6 = rate_fit["rate_fn"], rate_fit["s6"]
    frames, meta = [], {}
    for T in treatment_temps:
        stage = effective_stage_trajectory(
            T, rate_fn, s6=s6, t_grid=t_grid, baseline_temp=baseline_temp,
            phase1_end=phase1_end, shiftback_h=shiftback_h)
        frames.append(pd.DataFrame({"treatment_T": T, "real_time_h": t_grid,
                                    "eff_stage": stage}))
        i_sb = int(np.argmin(np.abs(t_grid - shiftback_h)))
        meta[T] = {"stage_at_shiftback": float(stage[i_sb]),
                   "rate_treatment": float(rate_fn(T))}
    return pd.concat(frames, ignore_index=True), meta


def always_at_temperature_trajectory(temp, rate_fit, *, t_grid,
                                     baseline_temp=BASELINE_TEMP,
                                     phase1_end=BASELINE_START_H):
    """Reference trajectory for an embryo held CONTINUOUSLY at ``temp`` from
    ``phase1_end`` onward (28C only for the first 6h, like every cohort).  For
    temp==28C this is the no-shift control; for temp==T it is the continuous-
    exposure sibling (what the measured hotfish cohorts are).  No shift-back."""
    t_grid = np.asarray(t_grid, dtype=float)
    rate_fn, s6 = rate_fit["rate_fn"], rate_fit["s6"]
    Tvec = np.where(t_grid < phase1_end, baseline_temp, temp)
    rates = np.array([rate_fn(T) for T in Tvec])
    dt = np.diff(t_grid, prepend=t_grid[0])
    cum = np.cumsum(rates * dt)
    i6 = int(np.argmin(np.abs(t_grid - phase1_end)))
    return s6 + (cum - cum[i6])


def equivalent_reference_age(target_stage, ref_time, ref_stage):
    """Given a target effective stage and a reference trajectory (monotonic
    ``ref_stage`` vs ``ref_time``), return the reference TIME at which the
    reference reaches ``target_stage`` -- i.e. the developmental-age equivalent.

    This is what lets 'shifted @ 24h' be matched to 'reference @ 30h': we invert
    the reference stage(time) curve.  Returns np.nan if target is out of range
    (extrapolation would be required)."""
    ref_time = np.asarray(ref_time, dtype=float)
    ref_stage = np.asarray(ref_stage, dtype=float)
    order = np.argsort(ref_stage)
    rs, rt = ref_stage[order], ref_time[order]
    if target_stage < rs.min() or target_stage > rs.max():
        return float("nan")
    return float(np.interp(target_stage, rs, rt))


def equivalent_age_table(treatment_temps, rate_fit, *, collection_times=(24, 30, 36),
                         shiftback_h=22.0, baseline_temp=BASELINE_TEMP,
                         phase1_end=BASELINE_START_H, t_max=60.0, n_grid=1201):
    """For every (treatment T, collection time), report the shifted cohort's
    effective stage at collection and the EQUIVALENT REFERENCE AGE against two
    references: always-28C and always-at-T (continuous exposure).

    Returns a long-form DataFrame: one row per (treatment_T, collection_h) with
    the shifted stage, the 28C-equivalent age, and the own-temp-equivalent age.
    'Equivalent age' = reference clock time whose effective stage matches the
    shifted embryo's -- so a value > collection time means the shifted embryo
    looks developmentally OLDER than that reference at the same clock time."""
    t_grid = np.linspace(0, t_max, n_grid)
    rate_fn, s6 = rate_fit["rate_fn"], rate_fit["s6"]
    ref28_stage = always_at_temperature_trajectory(baseline_temp, rate_fit, t_grid=t_grid)
    rows = []
    for T in treatment_temps:
        shifted = effective_stage_trajectory(
            T, rate_fn, s6=s6, t_grid=t_grid, baseline_temp=baseline_temp,
            phase1_end=phase1_end, shiftback_h=shiftback_h)
        refT_stage = always_at_temperature_trajectory(T, rate_fit, t_grid=t_grid)
        for ct in collection_times:
            i = int(np.argmin(np.abs(t_grid - ct)))
            s = float(shifted[i])
            rows.append({
                "treatment_T": T,
                "collection_h": ct,
                "shifted_stage": round(s, 2),
                "equiv_age_28C": round(equivalent_reference_age(s, t_grid, ref28_stage), 2),
                "equiv_age_ownT": round(equivalent_reference_age(s, t_grid, refT_stage), 2),
                "ref28_stage_at_collection": round(float(ref28_stage[i]), 2),
                "refT_stage_at_collection": round(float(refT_stage[i]), 2),
            })
    return pd.DataFrame(rows)


def stage_at_times(traj_df, real_times):
    """Table of effective stage at requested real-hpf for each cohort."""
    out = []
    for T, sub in traj_df.groupby("treatment_T"):
        row = {"treatment_T": T}
        for rt in real_times:
            i = int(np.argmin(np.abs(sub["real_time_h"].to_numpy() - rt)))
            row[f"{rt:g}h"] = round(float(sub["eff_stage"].to_numpy()[i]), 2)
        out.append(row)
    return pd.DataFrame(out)
