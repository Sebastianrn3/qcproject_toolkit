from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HARTREE_TO_KCALMOL = 627.5094740631


def standard_neb_muted(state: dict[str, Any]) -> bool:
    return bool(
        state.get("mute_standard_neb", False)
        or state.get("mute_standart_neb", False)
        or state.get("mute_all_neb", False)
        or state.get("prepare_guesses_only", False)
    )


def standard_run_name(state: dict[str, Any]) -> str:
    if state.get("standard_neb_native_name", True):
        return "standard"

    h = state.get("config_hash", "nohash")
    return f"standard_{h}"


def legacy_standard_run_name(state: dict[str, Any]) -> str | None:
    if "base_name" not in state or "pair_tag" not in state:
        return None

    if state.get("standard_neb_native_name", True):
        return f"{state['base_name']}_standard_{state['pair_tag']}"

    h = state.get("config_hash", "nohash")
    return f"{state['base_name']}_standard_{state['pair_tag']}_{h}"


def standard_run_name_candidates(state: dict[str, Any]) -> list[str]:
    candidates = []

    if state.get("standard_neb_run_name"):
        candidates.append(str(state["standard_neb_run_name"]))

    candidates.append(standard_run_name(state))

    legacy = legacy_standard_run_name(state)
    if legacy is not None:
        candidates.append(legacy)

    out = []
    seen = set()

    for name in candidates:
        if name in seen:
            continue

        seen.add(name)
        out.append(name)

    return out


def standard_result_json_candidates(state: dict[str, Any]) -> list[Path]:
    neb_folder = Path(state["base_cfg"].neb_folder)

    return [
        neb_folder / f"neb_{run_name}" / f"{run_name}_results.json"
        for run_name in standard_run_name_candidates(state)
    ]


def find_standard_result_json(state: dict[str, Any]) -> tuple[str | None, Path | None]:
    for run_name, result_json in zip(
        standard_run_name_candidates(state),
        standard_result_json_candidates(state),
    ):
        if result_json.exists():
            return run_name, result_json

    return None, None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data: dict[str, Any], *keys: str, default=None):
    cur = data

    for key in keys:
        if not isinstance(cur, dict):
            return default

        if key not in cur:
            return default

        cur = cur[key]

    return cur


def _get_fmax(block: dict[str, Any]) -> float:
    if not isinstance(block, dict):
        return 999.0

    if "neb_fmax" in block:
        return _safe_float(block["neb_fmax"], 999.0)

    if "fmax" in block:
        return _safe_float(block["fmax"], 999.0)

    return 999.0


def classify_standard_neb(state: dict[str, Any]) -> dict[str, Any]:
    if standard_neb_muted(state):
        return {
            "level": "muted",
            "budget": "normal",
            "standard_neb_muted": True,
            "reason": "standard NEB is muted",
        }

    run_name, result_json = find_standard_result_json(state)

    if result_json is None:
        checked = [str(p) for p in standard_result_json_candidates(state)]

        return {
            "level": "unknown",
            "budget": "normal",
            "standard_neb_muted": False,
            "reason": "standard NEB results not found",
            "checked": checked,
        }

    r = _read_json(result_json)

    cycles = _safe_int(r.get("n_neb_cycles", r.get("nsteps_taken", 0)), 0)
    converged = bool(r.get("converged", False))

    initial_block = r.get("initial", {})
    final_block = r.get("final", {})

    final_fmax = _get_fmax(final_block)

    initial_ea = _safe_float(
        _get_nested(r, "initial", "activation", default=0.0),
        0.0,
    ) * HARTREE_TO_KCALMOL

    final_ea = _safe_float(
        _get_nested(r, "final", "activation", default=0.0),
        0.0,
    ) * HARTREE_TO_KCALMOL

    ea_drop = initial_ea - final_ea

    if converged and cycles < 150 and final_fmax < 0.05 and initial_ea < 100:
        level = "easy"
        budget = "cheap"
    elif (not converged) or cycles > 500 or final_fmax > 0.15 or initial_ea > 500:
        level = "hard"
        budget = "heavy"
    else:
        level = "medium"
        budget = "normal"

    return {
        "level": level,
        "budget": budget,
        "standard_neb_muted": False,
        "run_name": run_name,
        "result_json": str(result_json),
        "converged": converged,
        "cycles": cycles,
        "final_fmax": final_fmax,
        "initial_ea_kcal": initial_ea,
        "final_ea_kcal": final_ea,
        "ea_drop_kcal": ea_drop,
    }


def apply_dynamic_guess_budget(
    state: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    state = dict(state)

    if info.get("standard_neb_muted", False):
        print(">>> Standard NEB is muted; dynamic custom guess budget is not applied")
        return state

    if state.get("force_champion_arena", False):
        print(">>> force_champion_arena=True; dynamic budget will not override arena settings")
        return state

    budget = info.get("budget", "normal")

    if budget == "cheap":
        state.update({
            "brute_top_n": 50,
            "reparam_target_candidates": 5,
            "reparam_stage1_keep": 3,
            "reparam_full_finalists": 1,
            "reparam_scf_check": True,
        })

        print(">>> Custom guess budget: CHEAP")

    elif budget == "heavy":
        state.update({
            "brute_top_n": 300,
            "reparam_target_candidates": 15,
            "reparam_stage1_keep": 5,
            "reparam_full_finalists": 5,
            "reparam_scf_check": True,
            "degenerate_extra_zoom": 4,
        })

        print(">>> Custom guess budget: HEAVY")

    else:
        state.update({
            "brute_top_n": 100,
            "reparam_target_candidates": 10,
            "reparam_stage1_keep": 4,
            "reparam_full_finalists": 3,
            "reparam_scf_check": True,
        })

        print(">>> Custom guess budget: NORMAL")

    return state