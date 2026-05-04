from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _method_variants(base_keys: list[str]) -> list[str]:
    out = []

    for key in base_keys:
        out.extend([
            f"{key}_w1",
            f"{key}_w2",
            f"{key}_wave1",
            f"{key}_wave2",
        ])

    return out


BASE_HASH_KEYS = [
    "name",
    "base_name",
    "pair_tag",
    "custom_guess_methods",

    "interpolation_method",
    "n_interpolated",
    "target_neb_beads",

    "image_pool_selection",
    "exclude_ends",
    "brute_force_laps",
    "k_select_laps",
    "zoom_radius_laps",
    "brute_top_n",
    "even_window",
    "even_coeficient",
    "even_coefficient",
    "degenerate_extra_zoom",
    "degenerate_zero_fraction",
    "degenerate_max_unique",

    "quality_max_step_cv",
    "retry_even_window",
    "retry_even_coeficient",
    "retry_even_coefficient",
    "retry_degenerate_extra_zoom",

    "mute_standard_neb",
    "mute_standart_neb",
    "mute_all_neb",
    "prepare_guesses_only",
    "mute_concours",

    "relax_max_scf_calls",
    "relax_max_scf_calls_w2",

    "reparam_scf_check",
    "reparam_target_candidates",
    "reparam_diversity_min_diff",
    "reparam_stage1_keep",
    "reparam_energy_wildcards",
    "reparam_energy_wildcard_pool",
    "reparam_energy_wildcard_min_diff",
    "reparam_spike_threshold_kcal",
    "reparam_recompute_originals_for_final",

    "reparam_w_ea",
    "reparam_w_jump",
    "reparam_w_perp_stage1",
    "reparam_w_perp_stage2",
    "reparam_w_fmax_stage1",
    "reparam_w_fmax_stage2",

    "reparam_max_allowed_ea_kcal",
    "reparam_max_allowed_jump_kcal",
    "reparam_max_allowed_step_cv",
    "reparam_max_allowed_step_ratio",
    "reparam_max_allowed_wiggle_ratio",

    "reparam_missing_perp_default_evA",
    "reparam_missing_fmax_default_evA",
    "reparam_missing_perp_penalty",
    "reparam_missing_fmax_penalty",

    "reparam_assume_single_barrier",
    "reparam_max_major_peaks",
    "reparam_max_post_peak_uphill_kcal",
    "reparam_max_secondary_rebound_kcal",
    "reparam_max_pre_peak_downhill_kcal",
    "reparam_max_allowed_fmax_evA",

    "clean_min_keep_target",
    "clean_rescue_min_keep",
    "clean_rescue_max_jump",
    "max_force_threshold",
    "endpoint_relax_cycles",
    "endpoint_max_scf_calls",
    "endpoint_relax_stop_kcal",

    "standard_neb_native_name",
    "neb_spring_constant",
    "neb_method",
    "neb_optimizer",
    "neb_fmax",
    "neb_steps",

    "GALIMA_NUSIRASYNETI",
]


METHOD_SPECIFIC_BASE_KEYS = [
    "n_interpolated",
    "brute_force_laps",
    "k_select_laps",
    "zoom_radius_laps",
    "brute_top_n",
    "even_window",
    "even_coeficient",
    "even_coefficient",
    "degenerate_extra_zoom",
    "degenerate_zero_fraction",
    "degenerate_max_unique",
    "relax_max_scf_calls",
]


HASH_KEYS = BASE_HASH_KEYS + _method_variants(METHOD_SPECIFIC_BASE_KEYS)


CFG_HASH_KEYS = [
    "charge",
    "unpaired_electrons",
    "fixed_atoms",
    "rigid_groups",
    "mopac_method",
    "method",
    "keywords",
    "mopac_keywords",
    "mopac_extra_keywords",
]


PATH_KEYS = [
    "xyz_start_raw",
    "xyz_end_raw",
    "xyz_start",
    "xyz_end",
    "xyz_int1_raw",
    "xyz_int1",
    "xyz_int2_raw",
    "xyz_int2",
    "xyz_int3_raw",
    "xyz_int3",
    "xyz_int4_raw",
    "xyz_int4",
]


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, set):
        return sorted(obj, key=str)

    if isinstance(obj, tuple):
        return list(obj)

    return str(obj)


def _normalize_for_json(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return _normalize_for_json(obj.tolist())

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(k): _normalize_for_json(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(x) for x in obj]

    if isinstance(obj, set):
        return [_normalize_for_json(x) for x in sorted(obj, key=str)]

    return obj


def _hash_payload(payload: dict[str, Any], length: int = 12) -> str:
    normalized = _normalize_for_json(payload)

    text = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stable_subset(
    state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in HASH_KEYS:
        if key in state:
            out[key] = state[key]

    cfg = state.get("base_cfg")

    if cfg is not None:
        for key in CFG_HASH_KEYS:
            if hasattr(cfg, key):
                out[f"cfg.{key}"] = getattr(cfg, key)

    for key in PATH_KEYS:
        if key in state and state[key] is not None:
            out[key] = str(state[key])

    if extra:
        out.update(extra)

    return _normalize_for_json(out)


def config_hash(
    state: dict[str, Any],
    extra: dict[str, Any] | None = None,
    length: int = 12,
) -> str:
    payload = stable_subset(state, extra=extra)
    return _hash_payload(payload, length=length)


def metadata_hash_extra(
    *,
    kind: str,
    method: str | None = None,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "method": method,
    }

    if run_name is not None:
        payload["run_name"] = run_name

    if extra:
        payload.update(extra)

    return payload


def file_fingerprint(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "mtime_ns": None,
            "size": None,
        }

    p = Path(path)

    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
            "mtime_ns": None,
            "size": None,
        }

    st = p.stat()

    return {
        "path": str(p),
        "exists": True,
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def write_metadata(
    folder: Path,
    *,
    state: dict[str, Any],
    kind: str,
    method: str | None = None,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    hash_extra = metadata_hash_extra(
        kind=kind,
        method=method,
        run_name=run_name,
        extra=extra,
    )

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "method": method,
        "run_name": run_name,
        "name": state.get("name"),
        "base_name": state.get("base_name"),
        "pair_tag": state.get("pair_tag"),
        "config_hash": config_hash(state, extra=hash_extra),
        "hash_extra": _normalize_for_json(hash_extra),
        "settings": stable_subset(state, extra=hash_extra),
    }

    path = folder / "metadata.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=_json_default)

    return path


def read_metadata(folder: Path) -> dict[str, Any] | None:
    path = Path(folder) / "metadata.json"

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read metadata: {path}\n       {exc}")
        return None


def metadata_expected_hash(
    *,
    state: dict[str, Any],
    kind: str,
    method: str | None = None,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    hash_extra = metadata_hash_extra(
        kind=kind,
        method=method,
        run_name=run_name,
        extra=extra,
    )

    return config_hash(state, extra=hash_extra)


def metadata_matches(
    folder: Path,
    *,
    state: dict[str, Any],
    kind: str,
    method: str | None = None,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
    verbose: bool = False,
) -> bool:
    meta = read_metadata(folder)

    if meta is None:
        if verbose:
            print(f"[metadata] no metadata.json in {folder}")
        return False

    expected = metadata_expected_hash(
        state=state,
        kind=kind,
        method=method,
        run_name=run_name,
        extra=extra,
    )

    actual = meta.get("config_hash")
    ok = actual == expected

    if verbose and not ok:
        print(
            f"[metadata] mismatch in {folder}\n"
            f"  actual   = {actual}\n"
            f"  expected = {expected}\n"
            f"  kind     = {kind}\n"
            f"  method   = {method}\n"
            f"  run_name = {run_name}\n"
            f"  extra    = {_normalize_for_json(extra or {})}"
        )

    return ok