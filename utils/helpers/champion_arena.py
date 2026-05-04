import json
from pathlib import Path
from dataclasses import replace
from typing import Dict, Any

import numpy as np

from settings.config import ANGSTROM_PER_BOHR
from utils.io.run_1scf import main_mopac


HARTREE_TO_KCALMOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / ANGSTROM_PER_BOHR


# ======================================================================================
# BASIC HELPERS
# ======================================================================================

def finite_or_zero(x):
    """For printing only. Do not use this for scoring missing forces."""
    try:
        x = float(x)
    except Exception:
        return 0.0

    if not np.isfinite(x):
        return 0.0

    return x


def finite_or(x, default):
    """
    Safe finite helper for scoring.

    Unlike finite_or_zero(), this does not turn NaN/inf into an artificially good 0.0.
    """
    try:
        x = float(x)
    except Exception:
        return default

    if not np.isfinite(x):
        return default

    return x


def safe_float(x, default=None):
    try:
        x = float(x)
    except Exception:
        return default

    if not np.isfinite(x):
        return default

    return x


def safe_int(x, default=-1):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def valid_grad(g) -> bool:
    if g is None:
        return False

    arr = np.asarray(g, dtype=float)

    return arr.size > 0 and np.all(np.isfinite(arr))


def grad_fmax_ev_ang(grad_eh_bohr):
    """
    Per-atom maximum gradient norm in eV/Å.
    Assumes grad is in Eh/bohr.
    """
    grad = np.asarray(grad_eh_bohr, dtype=float) * HARTREE_BOHR_TO_EV_ANG
    return float(np.linalg.norm(grad, axis=1).max())


def combo_signature(entry: Dict[str, Any]) -> np.ndarray:
    sig = []

    for i, idx in enumerate(entry["combo"]):
        item = entry["pool"][i][int(idx)]

        sig.append(
            item.get("source_image")
            if item.get("source_image") is not None
            else item.get("k_in_file", int(idx))
        )

    return np.asarray(sig, dtype=int)


def signature_distance(a, b) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)

    if len(a) != len(b):
        return 1.0

    return float(np.mean(a != b))


def select_diverse_entries(entries, n=15, min_diff_fraction=0.35):
    entries = sorted(entries, key=lambda x: x["score"], reverse=True)

    selected = []
    signatures = []

    for entry in entries:
        sig = combo_signature(entry)

        if not selected:
            selected.append(entry)
            signatures.append(sig)
        elif all(signature_distance(sig, s) >= min_diff_fraction for s in signatures):
            selected.append(entry)
            signatures.append(sig)

        if len(selected) >= n:
            break

    if len(selected) < n:
        for entry in entries:
            if any(entry is selected_entry for selected_entry in selected):
                continue

            selected.append(entry)

            if len(selected) >= n:
                break

    return selected


def entry_items(entry):
    return [
        entry["pool"][i][int(idx)]
        for i, idx in enumerate(entry["combo"])
    ]


# ======================================================================================
# PATH DENSIFICATION
# ======================================================================================

def densify_to_target_keep_originals_with_mask(path, target_nodes):
    """
    Densifies path to target_nodes by inserting linear interpolated beads.
    Original beads are never removed.

    Returns:
        geoms_new,
        is_original_mask
    """
    path = np.asarray(path, dtype=float)
    n_old = len(path)

    if target_nodes == n_old:
        return path.copy(), np.ones(n_old, dtype=bool)

    if target_nodes < n_old:
        raise ValueError(
            f"target_nodes={target_nodes} < len(path)={n_old}. "
            f"For custom guess, do not remove old beads."
        )

    n_add = target_nodes - n_old

    seg_lengths = np.array([
        np.linalg.norm((path[i + 1] - path[i]).ravel())
        for i in range(n_old - 1)
    ])

    inserts = np.zeros(n_old - 1, dtype=int)

    for _ in range(n_add):
        score = seg_lengths / (inserts + 1)
        j = int(np.argmax(score))
        inserts[j] += 1

    out = []
    is_original = []

    for i in range(n_old - 1):
        a = path[i]
        b = path[i + 1]

        out.append(a)
        is_original.append(True)

        for k in range(1, inserts[i] + 1):
            t = k / (inserts[i] + 1)
            out.append((1 - t) * a + t * b)
            is_original.append(False)

    out.append(path[-1])
    is_original.append(True)

    return np.asarray(out), np.asarray(is_original, dtype=bool)


# ======================================================================================
# ENERGY, GEOMETRY AND MEP-LIKENESS METRICS
# ======================================================================================

def energy_shape_metrics(e_rel_kcal):
    """
    Minimal reliable energy metrics for a raw guess.

    We intentionally do not score peak count, post-peak uphill, secondary rebound,
    or pre-peak downhill. NEB/String can smooth these raw-profile wrinkles.
    """
    e = np.asarray(e_rel_kcal, dtype=float).ravel()

    if len(e) == 0:
        return {
            "ea_kcal": 0.0,
            "peak_idx": 0,
            "max_jump_kcal": 0.0,
            # Compatibility fields. Do not use them for score.
            "pre_peak_downhill_kcal": 0.0,
            "post_peak_uphill_kcal": 0.0,
            "secondary_rebound_kcal": 0.0,
            "n_major_peaks": 0,
        }

    if not np.all(np.isfinite(e)):
        e = np.nan_to_num(e, nan=1e6, posinf=1e6, neginf=-1e6)

    de = np.diff(e)

    return {
        "ea_kcal": float(np.max(e)),
        "peak_idx": int(np.argmax(e)),
        "max_jump_kcal": float(np.max(np.abs(de))) if len(de) else 0.0,
        # Compatibility fields. Do not use them for score.
        "pre_peak_downhill_kcal": 0.0,
        "post_peak_uphill_kcal": 0.0,
        "secondary_rebound_kcal": 0.0,
        "n_major_peaks": 0,
    }


def gradient_fmax_metrics(grads):
    values = [grad_fmax_ev_ang(g) for g in grads if valid_grad(g)]
    return {"max_fmax": float(np.max(values)) if values else np.nan}


def gradient_mep_alignment_metrics(geoms, grads):
    """
    Orthogonal force diagnostic.

    On a reasonable MEP-like path, the gradient should be mostly parallel to the
    local tangent. Large perpendicular force means NEB will spend effort dragging
    the band sideways, or the band can become unstable.
    """
    geoms = np.asarray(geoms, dtype=float)
    perp_fmax_values = []

    if len(geoms) < 3:
        return {"max_perp_fmax": np.nan}

    for i in range(1, len(geoms) - 1):
        if i >= len(grads) or not valid_grad(grads[i]):
            continue

        tangent = (geoms[i + 1] - geoms[i - 1]).ravel()
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm < 1e-12:
            continue

        tangent = tangent / tangent_norm
        g = np.asarray(grads[i], dtype=float)
        g_flat = g.ravel()

        g_parallel = np.dot(g_flat, tangent) * tangent
        g_perp = g_flat - g_parallel

        g_perp_atoms = g_perp.reshape(g.shape) * HARTREE_BOHR_TO_EV_ANG
        perp_fmax = float(np.max(np.linalg.norm(g_perp_atoms, axis=1)))
        perp_fmax_values.append(perp_fmax)

    return {"max_perp_fmax": float(np.max(perp_fmax_values)) if perp_fmax_values else np.nan}


def geometry_path_metrics(geoms):
    """
    Robust geometry diagnostics:
      - step_cv: uneven bead spacing;
      - max_step_ratio: single segment too long relative to mean;
      - wiggle_ratio: loop detector, used softly and carefully.

    wiggle_ratio can explode when endpoints are geometrically close, so it is
    disabled when end-to-end distance is too small relative to path length.
    """
    geoms = np.asarray(geoms, dtype=float)

    if len(geoms) < 2:
        return {
            "max_step": 0.0,
            "mean_step": 0.0,
            "step_cv": 0.0,
            "max_step_ratio": 0.0,
            "wiggle_ratio": 1.0,
        }

    steps = np.linalg.norm(
        np.diff(geoms, axis=0).reshape(len(geoms) - 1, -1),
        axis=1,
    )

    mean_step = float(np.mean(steps))
    max_step = float(np.max(steps))
    step_cv = float(np.std(steps) / (mean_step + 1e-12))
    max_step_ratio = float(max_step / (mean_step + 1e-12))

    path_length = float(np.sum(steps))
    end_to_end = float(np.linalg.norm((geoms[-1] - geoms[0]).ravel()))

    if path_length < 1e-12:
        wiggle_ratio = 1.0
    elif end_to_end < 0.25 * path_length:
        # Endpoints are close compared to the travelled path.
        # In such cases path_length/end_to_end is not a fair loop metric.
        wiggle_ratio = 1.0
    else:
        wiggle_ratio = float(path_length / (end_to_end + 1e-12))

    return {
        "max_step": max_step,
        "mean_step": mean_step,
        "step_cv": step_cv,
        "max_step_ratio": max_step_ratio,
        "wiggle_ratio": wiggle_ratio,
    }


def tangent_smoothness_metrics(geoms):
    """
    Kept for compatibility. Not used in the final score.
    """
    geoms = np.asarray(geoms, dtype=float)

    if len(geoms) < 3:
        return {
            "max_turn": 0.0,
            "mean_turn": 0.0,
            "turn_sum": 0.0,
        }

    segs = []

    for i in range(len(geoms) - 1):
        v = (geoms[i + 1] - geoms[i]).ravel()
        n = np.linalg.norm(v)

        if n > 1e-12:
            segs.append(v / n)

    if len(segs) < 2:
        return {
            "max_turn": 0.0,
            "mean_turn": 0.0,
            "turn_sum": 0.0,
        }

    turns = []

    for i in range(len(segs) - 1):
        cosang = float(np.dot(segs[i], segs[i + 1]))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        turns.append(1.0 - cosang)

    turns = np.asarray(turns, dtype=float)

    return {
        "max_turn": float(np.max(turns)),
        "mean_turn": float(np.mean(turns)),
        "turn_sum": float(np.sum(turns)),
    }


def mep_constraint_penalty(energy_m, geom_m, turn_m, align_m, grad_m, state):
    """
    Compatibility wrapper.

    Old code used huge hard penalties here. The new scoring keeps everything in
    score_mep_likeness() as soft, inspectable components.
    """
    return 0.0


def score_mep_likeness(
    geoms,
    energies_abs,
    grads,
    state,
    worst_spike_kcal=0.0,
    stage="stage2",
):
    """
    Physical soft badness for champion arena.

    Lower is better.

    The score is not a true energy. It is a pseudo-action-like badness:
      - Ea: prefer lower barriers, but with modest weight;
      - F_perp: main stability criterion for NEB/String;
      - Fmax: catches atom clashes and generally unstable beads;
      - max energy jump: catches poor bead resolution;
      - quadratic soft walls: activate only after physically suspicious limits.
    """
    energies_abs = np.asarray(energies_abs, dtype=float)

    bad_metrics = {
        "score": 1e6,
        "score_ea": 1e6,
        "score_perp": 1e6,
        "score_fmax": 1e6,
        "score_jump": 1e6,
        "penalty_sum": 1e6,
        "penalty_ea": 1e6,
        "penalty_jump": 1e6,
        "penalty_step_cv": 1e6,
        "penalty_step_ratio": 1e6,
        "penalty_wiggle": 1e6,
        "penalty_spike": 1e6,
        "penalty_edge_peak": 1e6,
        "ea_kcal": 1e6,
        "peak_idx": 0,
        "max_jump_kcal": 1e6,
        "pre_peak_downhill_kcal": 0.0,
        "post_peak_uphill_kcal": 0.0,
        "secondary_rebound_kcal": 0.0,
        "n_major_peaks": 0,
        "max_step": 1e6,
        "mean_step": 1e6,
        "step_cv": 1e6,
        "max_step_ratio": 1e6,
        "wiggle_ratio": 1e6,
        "max_turn": 0.0,
        "mean_turn": 0.0,
        "turn_sum": 0.0,
        "max_fmax": 1e6,
        "max_perp_fmax": 1e6,
        "raw_max_fmax": np.nan,
        "raw_max_perp_fmax": np.nan,
        "missing_fmax": True,
        "missing_perp_fmax": True,
        "hard_penalty": 0.0,
        "edge_peak_penalty": 0.0,
    }

    if len(energies_abs) == 0:
        return 1e6, bad_metrics

    e_rel_kcal = (energies_abs - energies_abs[0]) * HARTREE_TO_KCALMOL

    energy_m = energy_shape_metrics(e_rel_kcal)
    geom_m = geometry_path_metrics(geoms)
    turn_m = tangent_smoothness_metrics(geoms)
    align_m = gradient_mep_alignment_metrics(geoms, grads)
    grad_m = gradient_fmax_metrics(grads)

    Ea = finite_or(energy_m["ea_kcal"], 1e6)
    max_jump = finite_or(energy_m["max_jump_kcal"], 1e6)

    raw_perp_fmax = align_m["max_perp_fmax"]
    raw_max_fmax = grad_m["max_fmax"]

    missing_perp_fmax = not np.isfinite(raw_perp_fmax)
    missing_fmax = not np.isfinite(raw_max_fmax)

    perp_default = float(state.get("reparam_missing_perp_default_evA", 50.0))
    fmax_default = float(state.get("reparam_missing_fmax_default_evA", 80.0))

    perp_fmax = finite_or(raw_perp_fmax, perp_default)
    max_fmax = finite_or(raw_max_fmax, fmax_default)

    step_cv = geom_m["step_cv"]
    step_ratio = geom_m["max_step_ratio"]
    wiggle = geom_m["wiggle_ratio"]
    peak_idx = int(energy_m["peak_idx"])

    # ----------------------------------------------------------------------
    # Soft quadratic walls. These do not dominate until the path is physically
    # suspicious. They replace discontinuous +1000/+100000 hard penalties.
    # ----------------------------------------------------------------------
    max_allowed_ea = float(state.get("reparam_max_allowed_ea_kcal", 80.0))
    max_allowed_jump = float(state.get("reparam_max_allowed_jump_kcal", 20.0))

    step_cv_soft = float(state.get("reparam_max_allowed_step_cv", 0.7))
    step_ratio_soft = float(state.get("reparam_max_allowed_step_ratio", 3.0))
    wiggle_soft = float(state.get("reparam_max_allowed_wiggle_ratio", 2.0))
    spike_soft = float(state.get("reparam_spike_threshold_kcal", 10.0))

    penalty_ea = 0.5 * max(0.0, Ea - max_allowed_ea) ** 2
    penalty_jump = 1.0 * max(0.0, max_jump - max_allowed_jump) ** 2
    penalty_step_cv = 20.0 * max(0.0, step_cv - step_cv_soft) ** 2
    penalty_step_ratio = 20.0 * max(0.0, step_ratio - step_ratio_soft) ** 2
    penalty_wiggle = 50.0 * max(0.0, wiggle - wiggle_soft) ** 2
    penalty_spike = 2.0 * max(0.0, worst_spike_kcal - spike_soft) ** 2

    penalty_edge_peak = 0.0
    if Ea > 5.0 and (peak_idx <= 1 or peak_idx >= len(energies_abs) - 2):
        penalty_edge_peak = 15.0

    penalty_missing = 0.0
    if missing_perp_fmax:
        penalty_missing += float(state.get("reparam_missing_perp_penalty", 50.0))
    if missing_fmax:
        penalty_missing += float(state.get("reparam_missing_fmax_penalty", 25.0))

    penalty_sum = float(
        penalty_ea
        + penalty_jump
        + penalty_step_cv
        + penalty_step_ratio
        + penalty_wiggle
        + penalty_spike
        + penalty_edge_peak
        + penalty_missing
    )

    # ----------------------------------------------------------------------
    # Linear physical components.
    # ----------------------------------------------------------------------
    w_perp = float(state.get("reparam_w_perp_stage1", 10.0)) if stage == "stage1" else float(state.get("reparam_w_perp_stage2", 25.0))
    w_fmax = float(state.get("reparam_w_fmax_stage1", 2.0)) if stage == "stage1" else float(state.get("reparam_w_fmax_stage2", 4.0))
    w_jump = float(state.get("reparam_w_jump", 1.0))
    w_ea = float(state.get("reparam_w_ea", 0.5))

    score_ea = w_ea * Ea
    score_perp = w_perp * perp_fmax
    score_fmax = w_fmax * max_fmax
    score_jump = w_jump * max_jump

    score = float(score_ea + score_perp + score_fmax + score_jump + penalty_sum)

    metrics = {}
    metrics.update(energy_m)
    metrics.update(geom_m)
    metrics.update(turn_m)

    metrics["raw_max_perp_fmax"] = safe_float(raw_perp_fmax, default=None)
    metrics["raw_max_fmax"] = safe_float(raw_max_fmax, default=None)
    metrics["missing_perp_fmax"] = bool(missing_perp_fmax)
    metrics["missing_fmax"] = bool(missing_fmax)

    # Store the actual values used for scoring.
    metrics["max_perp_fmax"] = float(perp_fmax)
    metrics["max_fmax"] = float(max_fmax)

    metrics["score_ea"] = float(score_ea)
    metrics["score_perp"] = float(score_perp)
    metrics["score_fmax"] = float(score_fmax)
    metrics["score_jump"] = float(score_jump)

    metrics["penalty_ea"] = float(penalty_ea)
    metrics["penalty_jump"] = float(penalty_jump)
    metrics["penalty_step_cv"] = float(penalty_step_cv)
    metrics["penalty_step_ratio"] = float(penalty_step_ratio)
    metrics["penalty_wiggle"] = float(penalty_wiggle)
    metrics["penalty_spike"] = float(penalty_spike)
    metrics["penalty_edge_peak"] = float(penalty_edge_peak)
    metrics["penalty_missing"] = float(penalty_missing)
    metrics["penalty_sum"] = float(penalty_sum)

    # Compatibility with older reports.
    metrics["hard_penalty"] = 0.0
    metrics["edge_peak_penalty"] = float(penalty_edge_peak)
    metrics["score"] = float(score)

    return float(score), metrics


# ======================================================================================
# SCF WRAPPER FOR REPARAM CHECKS
# ======================================================================================

def make_reparam_scf_func(atoms, cfg, workdir: Path, label_prefix: str):
    """
    Returns function:
        scf(geom_bohr) -> (energy_Eh, grad_Eh_bohr)

    Includes a cache by rounded geometry.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    counter = {"n": 0}
    cache = {}

    def key_geom(geom):
        return tuple(np.round(np.asarray(geom).ravel(), 8))

    def scf(geom_bohr):
        key = key_geom(geom_bohr)

        if key in cache:
            return cache[key]

        i = counter["n"]
        counter["n"] += 1

        local_dir = workdir / f"{label_prefix}_{i:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)

        local_cfg = replace(
            cfg,
            mopac_path=local_dir,
            jobname=f"{label_prefix}_{i:03d}",
        )

        e_eh, grad = main_mopac(atoms, geom_bohr, local_cfg)

        result = (float(e_eh), np.asarray(grad, dtype=float))
        cache[key] = result

        return result

    return scf


# ======================================================================================
# FAST ENERGY WILDCARDS
# ======================================================================================

def entry_energy_metrics(entry):
    """
    Fast preliminary energy estimate without new MOPAC calls.
    Used only for low-energy wildcards.
    """
    items = entry_items(entry)
    energies = np.asarray([x["energy"] for x in items], dtype=float)

    e_rel_kcal = (energies - energies[0]) * HARTREE_TO_KCALMOL
    energy_m = energy_shape_metrics(e_rel_kcal)

    return {
        "ea_kcal": float(energy_m["ea_kcal"]),
        "max_jump_kcal": float(energy_m["max_jump_kcal"]),
        "peak_idx": int(energy_m["peak_idx"]),
    }


def same_entry_object(a, b) -> bool:
    return a is b


def already_selected(entry, selected) -> bool:
    return any(same_entry_object(entry, x) for x in selected)


def select_low_energy_wildcards(
    entries,
    selected,
    n=5,
    pool_size=300,
    min_diff_fraction=0.25,
):
    """
    Takes several candidates with lowest preliminary Ea from the top brute-score pool,
    without duplicating already selected diverse entries.
    """
    by_score = sorted(entries, key=lambda x: x["score"], reverse=True)
    pool = by_score[:min(pool_size, len(by_score))]

    scored = []

    for entry in pool:
        if already_selected(entry, selected):
            continue

        metrics = entry_energy_metrics(entry)

        scored.append((
            metrics["ea_kcal"],
            metrics["max_jump_kcal"],
            entry,
            metrics,
        ))

    scored.sort(key=lambda x: (x[0], x[1]))

    out = []
    signatures = [combo_signature(e) for e in selected]

    for ea, jump, entry, metrics in scored:
        sig = combo_signature(entry)

        if signatures:
            ok_diverse = all(
                signature_distance(sig, s) >= min_diff_fraction
                for s in signatures
            )
        else:
            ok_diverse = True

        if not ok_diverse:
            continue

        entry["energy_wildcard_metrics"] = metrics
        out.append(entry)
        signatures.append(sig)

        if len(out) >= n:
            break

    return out


# ======================================================================================
# STAGE 1
# ======================================================================================

def stage1_energy_diagnostics(entry, atoms, cfg, state, method, rank):
    """
    Stage 1:
      - take candidate beads;
      - densify to target_neb_beads;
      - SCF E+grad only for inserted beads;
      - score with energy/geometry/partial-gradient diagnostics.
    """
    items = entry_items(entry)

    old_geoms = np.asarray([x["geom"] for x in items], dtype=float)
    old_energies = np.asarray([x["energy"] for x in items], dtype=float)
    old_grads = [x.get("grad") for x in items]

    geoms15, is_original = densify_to_target_keep_originals_with_mask(
        old_geoms,
        int(state["target_neb_beads"]),
    )

    energies15 = np.full(len(geoms15), np.nan)
    grads15 = [None] * len(geoms15)

    energies15[is_original] = old_energies

    old_i = 0

    for i in range(len(geoms15)):
        if is_original[i]:
            grads15[i] = old_grads[old_i]
            old_i += 1

    scf = make_reparam_scf_func(
        atoms=atoms,
        cfg=cfg,
        workdir=Path(cfg.analysis_folder) / "reparam_scf_check",
        label_prefix=f"{method}_stage1_cand{rank}",
    )

    inserted = np.where(~is_original)[0]

    for i in inserted:
        e, g = scf(geoms15[i])
        energies15[i] = e
        grads15[i] = g

    worst_spike = 0.0
    spike_records = []

    for i in inserted:
        left = i - 1

        while left >= 0 and not is_original[left]:
            left -= 1

        right = i + 1

        while right < len(is_original) and not is_original[right]:
            right += 1

        if left < 0 or right >= len(is_original):
            continue

        spike = (
            energies15[i] - max(energies15[left], energies15[right])
        ) * HARTREE_TO_KCALMOL

        worst_spike = max(worst_spike, float(spike))

        if spike > float(state.get("reparam_spike_threshold_kcal", 10.0)):
            spike_records.append({
                "bead": int(i),
                "spike_kcal": float(spike),
                "left": int(left),
                "right": int(right),
            })

    energy_badness, mep_metrics = score_mep_likeness(
        geoms=geoms15,
        energies_abs=energies15,
        grads=grads15,
        state=state,
        worst_spike_kcal=worst_spike,
        stage="stage1",
    )

    inserted_fmax = []

    for i in inserted:
        if valid_grad(grads15[i]):
            inserted_fmax.append(grad_fmax_ev_ang(grads15[i]))

    return {
        "entry": entry,
        "geoms": geoms15,
        "energies_abs": energies15,
        "grads": grads15,
        "is_original": is_original,
        "energy_badness": float(energy_badness),
        "ea_kcal": float(mep_metrics["ea_kcal"]),
        "max_jump_kcal": float(mep_metrics["max_jump_kcal"]),
        "worst_spike_kcal": float(worst_spike),
        "peak_idx": int(mep_metrics["peak_idx"]),
        "spike_records": spike_records,
        "inserted_max_fmax": float(max(inserted_fmax)) if inserted_fmax else np.nan,
        "raw_score": float(entry["score"]),
        "rank": rank,
        "mep_metrics": mep_metrics,
    }


# ======================================================================================
# STAGE 2
# ======================================================================================

def stage2_full_fmax_diagnostics(diag, atoms, cfg, state, method, finalist_rank):
    """
    Stage 2:
      - take best candidates after Stage 1;
      - recompute E+grad for original beads if enabled;
      - final score uses full fmax and MEP-like alignment.
    """
    geoms = np.asarray(diag["geoms"], dtype=float)
    energies = np.asarray(diag["energies_abs"], dtype=float).copy()
    grads = list(diag["grads"])
    is_original = np.asarray(diag["is_original"], dtype=bool)

    recompute_originals = bool(state.get("reparam_recompute_originals_for_final", True))

    scf = make_reparam_scf_func(
        atoms=atoms,
        cfg=cfg,
        workdir=Path(cfg.analysis_folder) / "reparam_scf_check",
        label_prefix=f"{method}_stage2_final{finalist_rank}",
    )

    if recompute_originals:
        indices_to_check = np.where(is_original)[0]
    else:
        indices_to_check = [
            i
            for i, g in enumerate(grads)
            if g is None or np.isnan(energies[i])
        ]

    for i in indices_to_check:
        e, g = scf(geoms[i])
        energies[i] = e
        grads[i] = g

    fmax_per_bead = []

    for g in grads:
        if valid_grad(g):
            fmax_per_bead.append(grad_fmax_ev_ang(g))
        else:
            fmax_per_bead.append(np.nan)

    fmax_per_bead = np.asarray(fmax_per_bead, dtype=float)

    if np.all(np.isnan(fmax_per_bead)):
        max_fmax = np.nan
    else:
        max_fmax = float(np.nanmax(fmax_per_bead))

    final_badness, mep_metrics = score_mep_likeness(
        geoms=geoms,
        energies_abs=energies,
        grads=grads,
        state=state,
        worst_spike_kcal=0.0,
        stage="stage2",
    )

    out = dict(diag)

    out.update({
        "energies_abs": energies,
        "grads": grads,
        "fmax_per_bead": fmax_per_bead,
        "max_fmax": max_fmax,
        "final_badness": float(final_badness),
        "final_ea_kcal": float(mep_metrics["ea_kcal"]),
        "final_max_jump_kcal": float(mep_metrics["max_jump_kcal"]),
        "final_peak_idx": int(mep_metrics["peak_idx"]),
        "mep_metrics": mep_metrics,
    })

    return out


# ======================================================================================
# PRINTING HELPERS
# ======================================================================================

def print_score_components(m, indent="  "):
    print(
        f"{indent}score = {finite_or_zero(m.get('score')):.4f}\n"
        f"{indent}components: "
        f"Ea={finite_or_zero(m.get('score_ea')):.3f}, "
        f"perp={finite_or_zero(m.get('score_perp')):.3f}, "
        f"fmax={finite_or_zero(m.get('score_fmax')):.3f}, "
        f"jump={finite_or_zero(m.get('score_jump')):.3f}, "
        f"penalty={finite_or_zero(m.get('penalty_sum')):.3f}\n"
        f"{indent}penalties: "
        f"Ea={finite_or_zero(m.get('penalty_ea')):.3f}, "
        f"jump={finite_or_zero(m.get('penalty_jump')):.3f}, "
        f"step_cv={finite_or_zero(m.get('penalty_step_cv')):.3f}, "
        f"step_ratio={finite_or_zero(m.get('penalty_step_ratio')):.3f}, "
        f"wiggle={finite_or_zero(m.get('penalty_wiggle')):.3f}, "
        f"spike={finite_or_zero(m.get('penalty_spike')):.3f}, "
        f"edge={finite_or_zero(m.get('penalty_edge_peak')):.3f}, "
        f"missing={finite_or_zero(m.get('penalty_missing')):.3f}"
    )


# ======================================================================================
# MAIN TOURNAMENT
# ======================================================================================

def choose_best_reparam_candidate(archive, atoms, cfg, state, method):
    """
    Main tournament:
      top archive
      -> diverse candidates
      -> optional low-energy wildcards
      -> Stage 1 soft MEP-like screening
      -> keep N finalists
      -> Stage 2 full fmax + MEP-like diagnostics
      -> choose 1 winner
    """
    diverse_n = int(state.get("reparam_target_candidates", 15))
    keep_n = int(state.get("reparam_stage1_keep", 5))
    min_diff = float(state.get("reparam_diversity_min_diff", 0.35))

    diverse = select_diverse_entries(
        archive,
        n=diverse_n,
        min_diff_fraction=min_diff,
    )

    energy_wildcards_n = int(state.get("reparam_energy_wildcards", 0))

    if energy_wildcards_n > 0:
        energy_wildcards = select_low_energy_wildcards(
            entries=archive,
            selected=diverse,
            n=energy_wildcards_n,
            pool_size=int(state.get("reparam_energy_wildcard_pool", 300)),
            min_diff_fraction=float(state.get("reparam_energy_wildcard_min_diff", 0.25)),
        )

        print(f"\n>>> Added low-energy wildcards for {method}: {len(energy_wildcards)}")

        for i, e in enumerate(energy_wildcards, start=1):
            m = e.get("energy_wildcard_metrics", {})

            print(
                f"  wildcard {i}: "
                f"raw_score={e['score']:.6f}, "
                f"Ea≈{m.get('ea_kcal', float('nan')):.3f} kcal/mol, "
                f"jump≈{m.get('max_jump_kcal', float('nan')):.3f} kcal/mol, "
                f"combo={e['combo']}"
            )

        diverse = diverse + energy_wildcards

    print(f"\n>>> Reparam tournament for {method}")
    print(f"    archive size: {len(archive)}")
    print(f"    candidates entering Stage 1: {len(diverse)}")
    print(f"    stage1 keep: {keep_n}")

    stage1 = []

    for rank, entry in enumerate(diverse, start=1):
        print(
            f"\n--- Stage 1 candidate {rank}/{len(diverse)} | "
            f"raw score={entry['score']:.6f} | lap={entry['lap'] + 1}"
        )

        diag = stage1_energy_diagnostics(
            entry=entry,
            atoms=atoms,
            cfg=cfg,
            state=state,
            method=method,
            rank=rank,
        )

        stage1.append(diag)

        m = diag["mep_metrics"]
        print_score_components(m)

        print(
            f"  Ea = {diag['ea_kcal']:.3f} kcal/mol\n"
            f"  max jump = {diag['max_jump_kcal']:.3f} kcal/mol\n"
            f"  worst inserted spike = {diag['worst_spike_kcal']:.3f} kcal/mol\n"
            f"  step_cv = {finite_or_zero(m.get('step_cv')):.3f}\n"
            f"  max step ratio = {finite_or_zero(m.get('max_step_ratio')):.3f}\n"
            f"  wiggle ratio = {finite_or_zero(m.get('wiggle_ratio')):.3f}\n"
            f"  max perp fmax used = {finite_or_zero(m.get('max_perp_fmax')):.3f} eV/Å"
            f" | raw = {m.get('raw_max_perp_fmax')}\n"
            f"  max fmax used = {finite_or_zero(m.get('max_fmax')):.3f} eV/Å"
            f" | raw = {m.get('raw_max_fmax')}\n"
            f"  inserted max fmax = {finite_or_zero(diag['inserted_max_fmax']):.3f} eV/Å\n"
            f"  missing perp/fmax = {m.get('missing_perp_fmax')} / {m.get('missing_fmax')}\n"
            f"  peak idx = {diag['peak_idx']}"
        )

    finalists = sorted(stage1, key=lambda d: d["energy_badness"])[:keep_n]

    print("\n>>> Stage 1 finalists:")

    for i, d in enumerate(finalists, start=1):
        m = d["mep_metrics"]

        print(
            f"  {i}. badness={d['energy_badness']:.4f}, "
            f"Ea={d['ea_kcal']:.3f}, "
            f"jump={d['max_jump_kcal']:.3f}, "
            f"perp={finite_or_zero(m.get('max_perp_fmax')):.3f}, "
            f"fmax={finite_or_zero(m.get('max_fmax')):.3f}, "
            f"step_cv={finite_or_zero(m.get('step_cv')):.3f}, "
            f"step_ratio={finite_or_zero(m.get('max_step_ratio')):.3f}, "
            f"wiggle={finite_or_zero(m.get('wiggle_ratio')):.3f}, "
            f"penalty={finite_or_zero(m.get('penalty_sum')):.1f}"
        )

    stage2 = []

    for i, diag in enumerate(finalists, start=1):
        print(f"\n--- Stage 2 finalist {i}/{len(finalists)}")

        full = stage2_full_fmax_diagnostics(
            diag=diag,
            atoms=atoms,
            cfg=cfg,
            state=state,
            method=method,
            finalist_rank=i,
        )

        stage2.append(full)

        m = full["mep_metrics"]
        print_score_components(m)

        print(
            f"  final Ea = {full['final_ea_kcal']:.3f} kcal/mol\n"
            f"  final max jump = {full['final_max_jump_kcal']:.3f} kcal/mol\n"
            f"  max fmax used = {finite_or_zero(m.get('max_fmax')):.3f} eV/Å"
            f" | raw = {m.get('raw_max_fmax')}\n"
            f"  max perp fmax used = {finite_or_zero(m.get('max_perp_fmax')):.3f} eV/Å"
            f" | raw = {m.get('raw_max_perp_fmax')}\n"
            f"  step_cv = {finite_or_zero(m.get('step_cv')):.3f}\n"
            f"  max step ratio = {finite_or_zero(m.get('max_step_ratio')):.3f}\n"
            f"  wiggle ratio = {finite_or_zero(m.get('wiggle_ratio')):.3f}\n"
            f"  fmax per bead = {np.array2string(full['fmax_per_bead'], precision=3)}"
        )

    best = min(stage2, key=lambda d: d["final_badness"])
    bm = best["mep_metrics"]

    print(
        f"\n>>> SELECTED {method} candidate:\n"
        f"  final_badness = {best['final_badness']:.4f}\n"
        f"  final Ea = {best['final_ea_kcal']:.3f} kcal/mol\n"
        f"  final max jump = {best['final_max_jump_kcal']:.3f} kcal/mol\n"
        f"  max fmax used = {finite_or_zero(bm.get('max_fmax')):.3f} eV/Å\n"
        f"  max perp fmax used = {finite_or_zero(bm.get('max_perp_fmax')):.3f} eV/Å\n"
        f"  penalty sum = {finite_or_zero(bm.get('penalty_sum')):.3f}\n"
        f"  raw brute score = {best['raw_score']:.6f}"
    )

    save_arena_report(
        cfg=cfg,
        method=method,
        diverse=diverse,
        stage1=stage1,
        finalists=finalists,
        stage2=stage2,
        best=best,
    )

    return best


# ======================================================================================
# REPORTING
# ======================================================================================

def metric_subset(m):
    if m is None:
        return {}

    keys = [
        "score",
        "score_ea",
        "score_perp",
        "score_fmax",
        "score_jump",
        "penalty_sum",
        "penalty_ea",
        "penalty_jump",
        "penalty_step_cv",
        "penalty_step_ratio",
        "penalty_wiggle",
        "penalty_spike",
        "penalty_edge_peak",
        "penalty_missing",
        "hard_penalty",
        "edge_peak_penalty",
        "ea_kcal",
        "max_jump_kcal",
        "pre_peak_downhill_kcal",
        "post_peak_uphill_kcal",
        "secondary_rebound_kcal",
        "n_major_peaks",
        "peak_idx",
        "max_step",
        "mean_step",
        "step_cv",
        "max_step_ratio",
        "wiggle_ratio",
        "max_turn",
        "mean_turn",
        "turn_sum",
        "max_fmax",
        "max_perp_fmax",
        "raw_max_fmax",
        "raw_max_perp_fmax",
    ]

    out = {}

    for k in keys:
        v = m.get(k)

        if isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = safe_float(v, default=None)

    out["missing_fmax"] = bool(m.get("missing_fmax", False))
    out["missing_perp_fmax"] = bool(m.get("missing_perp_fmax", False))

    return out


def save_arena_report(cfg, method, diverse, stage1, finalists, stage2, best):
    report_dir = Path(cfg.analysis_folder) / "champion_arena_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    txt_path = report_dir / f"{method}_arena_report.txt"
    json_path = report_dir / f"{method}_arena_report.json"

    def entry_origin_table(entry):
        rows = []

        for bead_i, idx in enumerate(entry["combo"]):
            item = entry["pool"][bead_i][int(idx)]

            rows.append({
                "bead": bead_i,
                "candidate_rank": int(idx),
                "source_file": item.get("source_file"),
                "source_image": safe_int(item.get("source_image", -1)),
                "source_n_images": safe_int(item.get("source_n_images", -1)),
                "energy_Eh": safe_float(item.get("energy", np.nan), default=None),
            })

        return rows

    data = {
        "method": method,
        "n_diverse": len(diverse),
        "n_stage1": len(stage1),
        "n_finalists": len(finalists),
        "diverse_candidates": [],
        "stage1": [],
        "stage2": [],
        "winner": {},
    }

    for i, entry in enumerate(diverse, start=1):
        data["diverse_candidates"].append({
            "candidate": i,
            "lap": int(entry["lap"]),
            "rank": int(entry["rank"]),
            "raw_score": float(entry["score"]),
            "combo": [int(x) for x in entry["combo"]],
            "origins": entry_origin_table(entry),
        })

    for i, d in enumerate(stage1, start=1):
        m = d.get("mep_metrics", {})

        data["stage1"].append({
            "candidate": i,
            "raw_score": float(d["raw_score"]),
            "energy_badness": float(d["energy_badness"]),
            "Ea_kcal": float(d["ea_kcal"]),
            "max_jump_kcal": float(d["max_jump_kcal"]),
            "worst_spike_kcal": float(d["worst_spike_kcal"]),
            "inserted_max_fmax": safe_float(d["inserted_max_fmax"], default=None),
            "peak_idx": int(d["peak_idx"]),
            "combo": [int(x) for x in d["entry"]["combo"]],
            "mep_metrics": metric_subset(m),
        })

    for i, d in enumerate(stage2, start=1):
        m = d.get("mep_metrics", {})

        energies_rel_kcal = (
            (np.asarray(d["energies_abs"], dtype=float) - float(d["energies_abs"][0]))
            * HARTREE_TO_KCALMOL
        )

        data["stage2"].append({
            "finalist": i,
            "raw_score": float(d["raw_score"]),
            "final_badness": float(d["final_badness"]),
            "final_Ea_kcal": float(d["final_ea_kcal"]),
            "final_max_jump_kcal": float(d["final_max_jump_kcal"]),
            "max_fmax": safe_float(d["max_fmax"], default=None),
            "final_peak_idx": int(d["final_peak_idx"]),
            "fmax_per_bead": [
                safe_float(x, default=None)
                for x in np.asarray(d["fmax_per_bead"], dtype=float)
            ],
            "energies_rel_kcal": energies_rel_kcal.tolist(),
            "combo": [int(x) for x in d["entry"]["combo"]],
            "mep_metrics": metric_subset(m),
        })

    winner_energies_rel_kcal = (
        (np.asarray(best["energies_abs"], dtype=float) - float(best["energies_abs"][0]))
        * HARTREE_TO_KCALMOL
    )

    data["winner"] = {
        "raw_score": float(best["raw_score"]),
        "final_badness": float(best["final_badness"]),
        "final_Ea_kcal": float(best["final_ea_kcal"]),
        "final_max_jump_kcal": float(best["final_max_jump_kcal"]),
        "max_fmax": safe_float(best["max_fmax"], default=None),
        "combo": [int(x) for x in best["entry"]["combo"]],
        "energies_rel_kcal": winner_energies_rel_kcal.tolist(),
        "fmax_per_bead": [
            safe_float(x, default=None)
            for x in np.asarray(best["fmax_per_bead"], dtype=float)
        ],
        "mep_metrics": metric_subset(best.get("mep_metrics", {})),
        "origins": entry_origin_table(best["entry"]),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"CHAMPION ARENA REPORT: {method}\n")
        f.write("=" * 90 + "\n\n")

        f.write("DIVERSE CANDIDATES\n")
        f.write("-" * 90 + "\n")

        for c in data["diverse_candidates"]:
            f.write(
                f"candidate {c['candidate']:02d} | "
                f"lap={c['lap'] + 1} | "
                f"rank={c['rank']} | "
                f"raw_score={c['raw_score']:.6f} | "
                f"combo={c['combo']}\n"
            )

            for row in c["origins"]:
                e = row["energy_Eh"]
                e_text = "None" if e is None else f"{e:.10f}"
                f.write(
                    f"  bead {row['bead']:02d}: "
                    f"rank={row['candidate_rank']:03d}, "
                    f"image={row['source_image'] + 1}/{row['source_n_images']}, "
                    f"E={e_text} Eh, "
                    f"file={row['source_file']}\n"
                )

            f.write("\n")

        f.write("\nSTAGE 1 MEP-LIKE SCREEN\n")
        f.write("-" * 90 + "\n")

        for s in data["stage1"]:
            m = s["mep_metrics"]

            f.write(
                f"candidate {s['candidate']:02d} | "
                f"badness={s['energy_badness']:.4f} | "
                f"Ea={s['Ea_kcal']:.3f} | "
                f"jump={s['max_jump_kcal']:.3f} | "
                f"score_Ea={finite_or_zero(m.get('score_ea')):.3f} | "
                f"score_perp={finite_or_zero(m.get('score_perp')):.3f} | "
                f"score_fmax={finite_or_zero(m.get('score_fmax')):.3f} | "
                f"score_jump={finite_or_zero(m.get('score_jump')):.3f} | "
                f"penalty={finite_or_zero(m.get('penalty_sum')):.1f} | "
                f"spike={s['worst_spike_kcal']:.3f} | "
                f"step_cv={finite_or_zero(m.get('step_cv')):.3f} | "
                f"step_ratio={finite_or_zero(m.get('max_step_ratio')):.3f} | "
                f"wiggle={finite_or_zero(m.get('wiggle_ratio')):.3f} | "
                f"perp_used={finite_or_zero(m.get('max_perp_fmax')):.3f} | "
                f"perp_raw={m.get('raw_max_perp_fmax')} | "
                f"fmax_used={finite_or_zero(m.get('max_fmax')):.3f} | "
                f"fmax_raw={m.get('raw_max_fmax')} | "
                f"inserted_fmax={finite_or_zero(s['inserted_max_fmax']):.3f} | "
                f"missing_perp={m.get('missing_perp_fmax')} | "
                f"missing_fmax={m.get('missing_fmax')} | "
                f"peak={s['peak_idx']}\n"
            )

        f.write("\nSTAGE 2 FINALISTS\n")
        f.write("-" * 90 + "\n")

        for s in data["stage2"]:
            m = s["mep_metrics"]

            f.write(
                f"finalist {s['finalist']:02d} | "
                f"final_badness={s['final_badness']:.4f} | "
                f"Ea={s['final_Ea_kcal']:.3f} | "
                f"jump={s['final_max_jump_kcal']:.3f} | "
                f"score_Ea={finite_or_zero(m.get('score_ea')):.3f} | "
                f"score_perp={finite_or_zero(m.get('score_perp')):.3f} | "
                f"score_fmax={finite_or_zero(m.get('score_fmax')):.3f} | "
                f"score_jump={finite_or_zero(m.get('score_jump')):.3f} | "
                f"penalty={finite_or_zero(m.get('penalty_sum')):.1f} | "
                f"step_cv={finite_or_zero(m.get('step_cv')):.3f} | "
                f"step_ratio={finite_or_zero(m.get('max_step_ratio')):.3f} | "
                f"wiggle={finite_or_zero(m.get('wiggle_ratio')):.3f} | "
                f"perp_used={finite_or_zero(m.get('max_perp_fmax')):.3f} | "
                f"perp_raw={m.get('raw_max_perp_fmax')} | "
                f"fmax_used={finite_or_zero(m.get('max_fmax')):.3f} | "
                f"fmax_raw={m.get('raw_max_fmax')} | "
                f"combo={s['combo']}\n"
            )

            f.write(
                f"  energies_rel_kcal="
                f"{np.array2string(np.array(s['energies_rel_kcal']), precision=3)}\n"
            )
            f.write(
                f"  fmax_per_bead="
                f"{np.array2string(np.array([finite_or_zero(x) for x in s['fmax_per_bead']]), precision=3)}\n"
            )

        f.write("\nWINNER\n")
        f.write("-" * 90 + "\n")

        w = data["winner"]
        m = w["mep_metrics"]

        f.write(
            f"final_badness={w['final_badness']:.4f}\n"
            f"Ea={w['final_Ea_kcal']:.3f} kcal/mol\n"
            f"max_jump={w['final_max_jump_kcal']:.3f} kcal/mol\n"
            f"score_Ea={finite_or_zero(m.get('score_ea')):.3f}\n"
            f"score_perp={finite_or_zero(m.get('score_perp')):.3f}\n"
            f"score_fmax={finite_or_zero(m.get('score_fmax')):.3f}\n"
            f"score_jump={finite_or_zero(m.get('score_jump')):.3f}\n"
            f"penalty_sum={finite_or_zero(m.get('penalty_sum')):.1f}\n"
            f"penalty_ea={finite_or_zero(m.get('penalty_ea')):.3f}\n"
            f"penalty_jump={finite_or_zero(m.get('penalty_jump')):.3f}\n"
            f"penalty_step_cv={finite_or_zero(m.get('penalty_step_cv')):.3f}\n"
            f"penalty_step_ratio={finite_or_zero(m.get('penalty_step_ratio')):.3f}\n"
            f"penalty_wiggle={finite_or_zero(m.get('penalty_wiggle')):.3f}\n"
            f"penalty_spike={finite_or_zero(m.get('penalty_spike')):.3f}\n"
            f"penalty_edge_peak={finite_or_zero(m.get('penalty_edge_peak')):.3f}\n"
            f"penalty_missing={finite_or_zero(m.get('penalty_missing')):.3f}\n"
            f"max_fmax_used={finite_or_zero(m.get('max_fmax')):.3f} eV/A\n"
            f"max_fmax_raw={m.get('raw_max_fmax')} eV/A\n"
            f"max_perp_fmax_used={finite_or_zero(m.get('max_perp_fmax')):.3f} eV/A\n"
            f"max_perp_fmax_raw={m.get('raw_max_perp_fmax')} eV/A\n"
            f"step_cv={finite_or_zero(m.get('step_cv')):.3f}\n"
            f"max_step_ratio={finite_or_zero(m.get('max_step_ratio')):.3f}\n"
            f"wiggle_ratio={finite_or_zero(m.get('wiggle_ratio')):.3f}\n"
            f"combo={w['combo']}\n"
        )

        f.write(
            f"energies_rel_kcal="
            f"{np.array2string(np.array(w['energies_rel_kcal']), precision=3)}\n"
        )
        f.write(
            f"fmax_per_bead="
            f"{np.array2string(np.array([finite_or_zero(x) for x in w['fmax_per_bead']]), precision=3)}\n"
        )

    print(f">>> Arena report saved:\n  {txt_path}\n  {json_path}")