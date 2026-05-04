from __future__ import annotations

from dataclasses import replace
from typing import Any

from utils.defaults import DEFAULT_BRUTE, DEFAULT_NEB, DEFAULT_RELAX
from utils.helpers.metadata import config_hash


def normalize_even_coefficient(state: dict[str, Any]) -> float:
    if "even_coefficient" in state:
        coef = state["even_coefficient"]
    elif "even_coeficient" in state:
        coef = state["even_coeficient"]
    else:
        coef = 1.0

    coef = float(coef)

    state["even_coefficient"] = coef
    state["even_coeficient"] = coef

    return coef


def normalize_runtime_flags(state: dict[str, Any]) -> None:
    if bool(state.get("prepare_guesses_only", False)):
        state["mute_all_neb"] = True

    if bool(state.get("mute_all_neb", False)):
        state["mute_standard_neb"] = True
        state["mute_standart_neb"] = True


def format_number_for_name(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x).replace(".", "p")

    if x.is_integer():
        return str(int(x))

    return f"{x:g}".replace(".", "p")


def make_method_suffix(state: dict[str, Any]) -> str:
    window = state.get("even_window", 1)
    coef = normalize_even_coefficient(state)

    window_s = format_number_for_name(window)
    coef_s = format_number_for_name(coef)

    return f"w{window_s}c{coef_s}"


def make_base_name(prefix: str, interpolation: str, state: dict[str, Any]) -> str:
    return f"{prefix}_{make_method_suffix(state)}_{interpolation}"


def make_pipeline_config(
    *,
    job,
    base_run_name: str,
    pair_tag: str,
    start_raw,
    end_raw,
    interpolation: str = "linear",
    overrides: dict[str, Any] | None = None,
):
    overrides = overrides or {}

    state: dict[str, Any] = {}
    state.update(DEFAULT_BRUTE)
    state.update(DEFAULT_NEB)
    state.update(DEFAULT_RELAX)
    state.update(overrides)

    normalize_even_coefficient(state)
    normalize_runtime_flags(state)

    base_name = make_base_name(base_run_name, interpolation, state)
    pair_name = f"{base_name}_{pair_tag}"

    geometries_folder = job.CFG.geometries_folder / pair_name
    opt_folder = job.CFG.opt_folder / pair_name
    neb_folder = job.CFG.neb_folder / pair_name

    cfg = replace(
        job.CFG,
        geometries_folder=geometries_folder,
        opt_folder=opt_folder,
        neb_folder=neb_folder,
        analysis_folder=neb_folder,
    )

    for folder in [
        cfg.geometries_folder,
        cfg.opt_folder,
        cfg.neb_folder,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    job.CFG.inputs_folder.mkdir(parents=True, exist_ok=True)

    state.update({
        "name": pair_name,
        "base_name": base_name,
        "pair_tag": pair_tag,
        "base_cfg": cfg,
        "interpolation_method": interpolation,

        "xyz_start_raw": start_raw,
        "xyz_end_raw": end_raw,
        "xyz_start": job.CFG.inputs_folder / f"{pair_name}_start_prepared.xyz",
        "xyz_end": job.CFG.inputs_folder / f"{pair_name}_end_prepared.xyz",
    })

    state["config_hash"] = config_hash(state)

    print(f"Config hash: {state['config_hash']}")

    return state