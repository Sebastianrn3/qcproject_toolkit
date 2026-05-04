from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils.brute_multiscan.pull_builder import build_image_groups, build_secondary_ranges_from_best
from utils.brute_multiscan.relax_images import (
    run_and_record_given_images_relaxation,
    run_and_record_interpolated_images_relaxation,
)
from utils.brute_multiscan.score_bruteforce import brute_force_trajectories
from utils.helpers.champion_arena import (
    choose_best_reparam_candidate,
    densify_to_target_keep_originals_with_mask,
)
from utils.helpers.metadata import file_fingerprint, metadata_matches, write_metadata
from utils.helpers.naming import make_pipeline_config
from utils.io.npz_io import load_all_npz_dict
from utils.io.xyz_io import write_xyz_series
from utils.pipeline.ase_neb import HARTREE_TO_KCALMOL, neb_wrapper
from utils.pipeline.endpoint_optim import stage_0_align
from utils.statistics.clasify import apply_dynamic_guess_budget, classify_standard_neb
from utils.statistics.clean_pool_outliers import regenerate_raw_npz_to_clean
from utils.statistics.plotgen import compare_all_neb


SUPPORTED_METHODS = {"wave1", "wave2"}


def concours_enabled(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("reparam_scf_check", True)
        and not state.get("mute_concours", False)
    )


def guess_variant(state: Dict[str, Any]) -> str:
    return "concoursed" if concours_enabled(state) else "basic"


def guess_method_label(state: Dict[str, Any], method: str) -> str:
    if method.endswith("basic") or method.endswith("concoursed"):
        return method

    return f"{method}{guess_variant(state)}"


def standard_neb_muted(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("mute_standard_neb", False)
        or state.get("mute_standart_neb", False)
        or all_neb_muted(state)
    )


def all_neb_muted(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("mute_all_neb", False)
        or state.get("prepare_guesses_only", False)
    )


def standard_run_name(state: Dict[str, Any]) -> str:
    if state.get("standard_neb_native_name", True):
        return "standard"

    h = state.get("config_hash", "nohash")
    return f"standard_{h}"


def custom_run_name(state: Dict[str, Any], method_name: str) -> str:
    return method_name


def wave_neb_folder(state: Dict[str, Any], method: str) -> Path:
    method_label = guess_method_label(state, method)
    run_name = custom_run_name(state, method_label)
    return state["base_cfg"].neb_folder / f"neb_{run_name}"


def cache_state_for_relax(state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)

    for key in [
        "mute_concours",
        "mute_all_neb",
        "prepare_guesses_only",
        "mute_standard_neb",
        "mute_standart_neb",
        "force_rebuild_custom_guess",
        "force_rebuild_wave1",
        "force_rebuild_wave2",
    ]:
        out.pop(key, None)

    return out


def has_files(path: Path) -> bool:
    path = Path(path)
    return path.exists() and any(path.iterdir())


def has_npz_files(path: Path) -> bool:
    path = Path(path)
    return path.exists() and path.is_dir() and any(path.glob("*.npz"))


def folder_is_valid_or_trusted(
    folder: Path,
    *,
    state: Dict[str, Any],
    kind: str,
    method: str,
    trust_key: str,
    extra: Dict[str, Any] | None = None,
) -> bool:
    folder = Path(folder)

    if not has_npz_files(folder):
        return False

    if metadata_matches(
        folder,
        state=state,
        kind=kind,
        method=method,
        extra=extra,
    ):
        return True

    if bool(state.get(trust_key, True)):
        print(
            f"[WARN] {method} {kind} folder has npz files but metadata is missing/mismatched.\n"
            f"       Trusting existing files because {trust_key}=True:\n"
            f"       {folder}"
        )
        return True

    return False


def lap_param(values: Sequence[Any], lap: int) -> Any:
    return values[min(lap, len(values) - 1)]


def get_even_coef(state: Dict[str, Any]) -> float:
    return float(state.get("even_coefficient", state.get("even_coeficient", 1.0)))


def get_method_param(
    state: Dict[str, Any],
    key: str,
    method: str,
    default: Any = None,
) -> Any:
    aliases = {
        "wave1": ("wave1", "w1"),
        "wave2": ("wave2", "w2"),
    }

    for alias in aliases.get(method, (method,)):
        method_key = f"{key}_{alias}"

        if method_key in state:
            return state[method_key]

    return state.get(key, default)


def make_method_state(state: Dict[str, Any], method: str) -> Dict[str, Any]:
    out = dict(state)

    out["n_interpolated"] = int(
        get_method_param(
            state,
            "n_interpolated",
            method,
            state["n_interpolated"],
        )
    )

    out["brute_force_laps"] = int(
        get_method_param(
            state,
            "brute_force_laps",
            method,
            state.get("brute_force_laps", 1),
        )
    )

    out["k_select_laps"] = list(
        get_method_param(
            state,
            "k_select_laps",
            method,
            state["k_select_laps"],
        )
    )

    out["zoom_radius_laps"] = list(
        get_method_param(
            state,
            "zoom_radius_laps",
            method,
            state["zoom_radius_laps"],
        )
    )

    out["relax_max_scf_calls"] = get_method_param(
        state,
        "relax_max_scf_calls",
        method,
        state.get("relax_max_scf_calls"),
    )

    return out


def geom_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()))


def chain_quality(geoms, energies_abs=None):
    geoms = np.asarray(geoms, dtype=float)

    steps = np.array([
        np.linalg.norm((geoms[i + 1] - geoms[i]).ravel())
        for i in range(len(geoms) - 1)
    ])

    mean_step = float(np.mean(steps)) if len(steps) else 0.0
    max_step = float(np.max(steps)) if len(steps) else 0.0
    step_cv = float(np.std(steps) / (mean_step + 1e-12)) if len(steps) else 0.0

    if energies_abs is None:
        ea = 0.0
        max_energy_jump = 0.0
        peak_idx = -1
        edge_peak_penalty = 0.0
    else:
        e = np.asarray(energies_abs, dtype=float)
        e_rel_kcal = (e - e[0]) * HARTREE_TO_KCALMOL

        ea = float(np.max(e_rel_kcal))
        max_energy_jump = float(np.max(np.abs(np.diff(e_rel_kcal)))) if len(e_rel_kcal) > 1 else 0.0
        peak_idx = int(np.argmax(e_rel_kcal))

        edge_peak_penalty = 0.0

        if peak_idx <= 1 or peak_idx >= len(e_rel_kcal) - 2:
            edge_peak_penalty = 10.0

    badness = (
        1.0 * ea
        + 0.5 * max_energy_jump
        + 10.0 * step_cv
        + edge_peak_penalty
    )

    return {
        "badness": float(badness),
        "ea_kcal": float(ea),
        "max_energy_jump_kcal": float(max_energy_jump),
        "step_cv": float(step_cv),
        "max_step": float(max_step),
        "peak_idx": int(peak_idx),
    }


def print_chain_quality(label, geoms, energies_abs=None):
    q = chain_quality(geoms, energies_abs)

    print(
        f"\n[{label} quality]\n"
        f"  badness              = {q['badness']:.4f}\n"
        f"  Ea initial, kcal/mol = {q['ea_kcal']:.4f}\n"
        f"  max E jump, kcal/mol = {q['max_energy_jump_kcal']:.4f}\n"
        f"  geometry step CV     = {q['step_cv']:.4f}\n"
        f"  max geom step        = {q['max_step']:.4f}\n"
        f"  peak index           = {q['peak_idx']}\n"
    )

    return q


def path_arc_positions(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)

    if len(path) <= 1:
        return np.array([0.0])

    flat = path.reshape(len(path), -1)
    dists = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    cum = np.insert(np.cumsum(dists), 0, 0.0)

    if cum[-1] < 1e-12:
        return np.linspace(0.0, 1.0, len(path))

    return cum / cum[-1]


def uniformly_resample_path(path, target_nodes):
    path = np.asarray(path, dtype=float)

    if len(path) == target_nodes:
        return path.copy()

    if target_nodes <= 2:
        return np.asarray([path[0], path[-1]], dtype=float)

    flat = path.reshape(len(path), -1)
    old_s = path_arc_positions(path)
    new_s = np.linspace(0.0, 1.0, target_nodes)

    out_flat = np.empty((target_nodes, flat.shape[1]), dtype=float)

    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(new_s, old_s, flat[:, j])

    return out_flat.reshape(target_nodes, *path.shape[1:])


def resample_values_on_path(path, values, target_nodes):
    values = np.asarray(values, dtype=float)

    if len(values) == target_nodes:
        return values.copy()

    old_s = path_arc_positions(path)
    new_s = np.linspace(0.0, 1.0, target_nodes)

    return np.interp(new_s, old_s, values)


def save_chain_summary(
    *,
    output_folder: Path,
    stem: str,
    atoms: Sequence[Any],
    geoms: Sequence[np.ndarray],
    energies_abs: Sequence[float],
) -> Dict[str, Any]:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    geometries_folder = output_folder / "geometries"
    geometries_folder.mkdir(parents=True, exist_ok=True)

    geoms = np.asarray(geoms, dtype=float)
    energies_abs = np.asarray(energies_abs, dtype=float)

    if len(geoms) != len(energies_abs):
        raise ValueError(
            f"Cannot save chain summary: len(geoms)={len(geoms)} "
            f"but len(energies_abs)={len(energies_abs)}"
        )

    energies_rel = energies_abs - np.min(energies_abs)
    npz_path = output_folder / f"{stem}.npz"
    quality = chain_quality(geoms, energies_abs)

    np.savez_compressed(
        npz_path,
        atoms=np.asarray(atoms),
        geoms=geoms,
        energies=energies_rel,
        energies_abs=energies_abs,
        dist_to_R=np.array([geom_dist(g, geoms[0]) for g in geoms]),
        dist_to_P=np.array([geom_dist(g, geoms[-1]) for g in geoms]),
        quality_badness=quality["badness"],
        quality_ea_kcal=quality["ea_kcal"],
        quality_max_energy_jump_kcal=quality["max_energy_jump_kcal"],
        quality_step_cv=quality["step_cv"],
        quality_peak_idx=quality["peak_idx"],
    )

    write_xyz_series(
        np.asarray(atoms),
        geoms,
        f"{stem}_geometries",
        geometries_folder,
        flatten_all_to_one=True,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(energies_rel)), energies_rel)
    plt.title(f"Energy Profile: {stem}")
    plt.ylabel("Relative Energy")
    plt.xlabel("Bead Index")
    plt.tight_layout()
    plt.savefig(output_folder / f"{stem}_energies.png", dpi=200)
    plt.close()

    print_chain_quality(stem, geoms, energies_abs)

    return {
        "npz": npz_path,
        "geoms": geoms,
        "energies_rel": energies_rel,
    }


def load_chain(npz_path: Path) -> Dict[str, Any]:
    keys = ["atoms", "geoms", "energies", "energies_abs", "dist_to_R", "dist_to_P"]

    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in keys if key in data}


def first_present(item: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if item.get(key) is not None:
            return item[key]

    return default


def format_origin(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "unknown source"

    source_file = first_present(
        item,
        ["source_file", "relaxation_file", "filename", "file", "path"],
        "unknown relaxation",
    )

    image = first_present(
        item,
        ["source_image", "frame_idx", "image_idx", "step_idx", "idx"],
    )

    n_images = first_present(
        item,
        ["source_n_images", "n_images", "n_frames", "total_frames", "trajectory_len"],
    )

    if image is None:
        return f"{source_file}, image unknown"

    image = int(image) + 1

    if n_images is None:
        return f"image {image} in {source_file}"

    return f"image {image} of {int(n_images)} in {source_file}"


def print_origins(
    title: str,
    pool: Sequence[Sequence[Dict[str, Any]]],
    combo: Sequence[int] | None,
) -> None:
    print(f"\n{title}\n{'-' * len(title)}")

    if combo is None:
        print("combo is None")
        return

    if len(pool) != len(combo):
        print(f"WARNING: len(pool)={len(pool)} != len(combo)={len(combo)}")

    for i, (candidates, idx) in enumerate(zip(pool, combo), start=1):
        idx = int(idx)

        if not 0 <= idx < len(candidates):
            print(f"bead {i}: INVALID candidate index {idx}; available 0..{len(candidates) - 1}")
            continue

        item = candidates[idx]
        energy = item.get("energy") if isinstance(item, dict) else None
        energy = f", energy = {float(energy):.8f}" if energy is not None else ""

        print(f"bead {i:02d}: candidate {idx:03d} -> {format_origin(item)}{energy}")


def is_degenerate_combo(
    combo: Sequence[int],
    max_unique: int = 2,
    zero_fraction: float = 0.75,
) -> bool:
    combo = np.asarray(combo, dtype=int)

    if len(combo) == 0:
        return False

    frac_zero = np.mean(combo == 0)
    n_unique = len(set(combo.tolist()))

    return frac_zero >= zero_fraction or n_unique <= max_unique


def brute_search(state: Dict[str, Any], raw_folder: Path, clean_folder: Path) -> Dict[str, Any]:
    best = {
        "score": -float("inf"),
        "combo": None,
        "pool": None,
        "p_idx": None,
    }

    archive = []
    loosen_next = False
    top_n = int(state.get("brute_top_n", 300))

    for lap in range(int(state.get("brute_force_laps", 1))):
        k = int(lap_param(state["k_select_laps"], lap))

        if lap == 0:
            pool, p_idx = build_image_groups(
                clean_folder,
                k,
                state["image_pool_selection"],
                exclude_ends=state["exclude_ends"],
            )
        else:
            if best["combo"] is None:
                raise ValueError("Cannot zoom: first brute-force lap found no chain.")

            z_r = int(lap_param(state["zoom_radius_laps"], lap - 1))

            if loosen_next:
                z_r += int(state.get("degenerate_extra_zoom", 2))
                print(f"[!] Loosening zoom radius on lap {lap + 1}: radius={z_r}")

            ranges = build_secondary_ranges_from_best(
                best["combo"],
                best["p_idx"],
                raw_folder,
                k,
                z_r,
            )

            pool, p_idx = build_image_groups(
                raw_folder,
                k,
                state["image_pool_selection"],
                ranges=ranges,
                exclude_ends=0,
            )

        results = brute_force_trajectories(
            pool,
            get_even_coef(state),
            state["even_window"],
            top_n=top_n,
        )

        result_list = results if isinstance(results, list) else [results]

        for rank, cand in enumerate(result_list, start=1):
            combo = [int(x) for x in cand["full_combo"]]
            score = float(cand["total_score"])

            archive.append({
                "lap": lap,
                "rank": rank,
                "score": score,
                "combo": combo,
                "candidate": cand,
                "pool": pool,
                "p_idx": p_idx,
            })

        candidate = result_list[0]
        combo = [int(x) for x in candidate["full_combo"]]
        score = float(candidate["total_score"])

        degenerate = is_degenerate_combo(
            combo,
            max_unique=int(state.get("degenerate_max_unique", 2)),
            zero_fraction=float(state.get("degenerate_zero_fraction", 0.75)),
        )

        if degenerate:
            print(f"[!] Degenerate combo on lap {lap + 1}: {combo}")
            print(f"    zero fraction = {np.mean(np.asarray(combo) == 0):.2f}")
            print(f"    unique ranks = {sorted(set(combo))}")

        loosen_next = degenerate

        if score >= best["score"]:
            best = {
                "score": score,
                "combo": combo,
                "pool": pool,
                "p_idx": p_idx,
                "lap": lap,
                "rank": 1,
            }

    if best["combo"] is None:
        raise ValueError("Brute-force search failed: no best chain was found.")

    best["archive"] = archive

    print(f"\nBrute-force archive collected: {len(archive)} candidates")

    return best


def best_items(best: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        best["pool"][i][idx]
        for i, idx in enumerate(best["combo"])
    ]


def make_basic_guess_from_best(best: Dict[str, Any], target_nodes: int):
    items = best_items(best)

    geoms = np.asarray([x["geom"] for x in items], dtype=float)
    energies_abs = np.asarray([x["energy"] for x in items], dtype=float)

    if len(geoms) != target_nodes:
        geoms_new = uniformly_resample_path(geoms, target_nodes)
        energies_new = resample_values_on_path(geoms, energies_abs, target_nodes)
        return geoms_new, energies_new

    return geoms, energies_abs


def run_interpolated_relaxation_with_optional_rigid_groups(
    state: Dict[str, Any],
    cfg: Any,
    raw_folder: Path,
) -> None:
    kwargs = dict(
        npz_subfolder=raw_folder.name,
        max_scf_calls=state.get("relax_max_scf_calls"),
    )

    rigid_groups = getattr(cfg, "rigid_groups", None)

    if rigid_groups is not None:
        kwargs["rigid_groups"] = rigid_groups

    try:
        run_and_record_interpolated_images_relaxation(
            state["xyz_start"],
            state["xyz_end"],
            state["interpolation_method"],
            state["n_interpolated"],
            cfg,
            cfg.fixed_atoms,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("rigid_groups", None)
        run_and_record_interpolated_images_relaxation(
            state["xyz_start"],
            state["xyz_end"],
            state["interpolation_method"],
            state["n_interpolated"],
            cfg,
            cfg.fixed_atoms,
            **kwargs,
        )


def run_given_relaxation_with_optional_rigid_groups(
    *,
    atoms,
    geoms,
    state: Dict[str, Any],
    cfg: Any,
    raw_folder: Path,
) -> None:
    kwargs = dict(
        npz_subfolder=raw_folder.name,
        max_scf_calls=state.get("relax_max_scf_calls"),
    )

    rigid_groups = getattr(cfg, "rigid_groups", None)

    if rigid_groups is not None:
        kwargs["rigid_groups"] = rigid_groups

    try:
        run_and_record_given_images_relaxation(
            atoms,
            geoms,
            cfg,
            cfg.fixed_atoms,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("rigid_groups", None)
        run_and_record_given_images_relaxation(
            atoms,
            geoms,
            cfg,
            cfg.fixed_atoms,
            **kwargs,
        )


def run_wave_search(
    state: Dict[str, Any],
    method: str,
    source_npz: Path | None = None,
) -> Path | None:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown wave method: {method}")

    cfg = state["base_cfg"]
    method_state = make_method_state(state, method)

    method_label = guess_method_label(method_state, method)
    wave_folder = wave_neb_folder(method_state, method)
    wave_folder.mkdir(parents=True, exist_ok=True)

    method_state["wave_neb_folder"] = wave_folder
    method_state["arena_report_dir"] = wave_folder
    method_state["reparam_scf_check_dir"] = wave_folder / "reparam_scf_check"

    prefix = f"{state['name']}_{method}"

    if method == "wave1":
        stem = "wave1_primary_chain"
    else:
        stem = "wave2_best_chain"

    target_npz = wave_folder / f"{stem}.npz"

    if method == "wave1":
        state["wave1_primary_chain_npz"] = target_npz

    method_state["wave1_primary_chain_npz"] = state.get("wave1_primary_chain_npz")
    method_state["final_guess_npzs"] = state.get("final_guess_npzs", {})

    raw_folder = cfg.opt_folder / f"{prefix}_raw"
    clean_folder = cfg.opt_folder / f"{prefix}_clean"

    force_rebuild_guess = bool(state.get("force_rebuild_custom_guess", False))
    force_rebuild_this_method = bool(state.get(f"force_rebuild_{method}", False))

    if target_npz.exists() and not force_rebuild_this_method and not force_rebuild_guess:
        print(f"{method_label} already exists. Skipping.")
        state["final_guess_npzs"][method_label] = target_npz
        return target_npz

    if target_npz.exists():
        print(f"[FORCE] {method_label} exists, but rebuilding selection: {target_npz}")

    print(f"\n{'=' * 40}")
    print(f"=== {method_label.upper()} SEARCH ===")
    print(f"{'=' * 40}")
    print(f"output folder = {wave_folder}")
    print(f"n_interpolated = {method_state['n_interpolated']}")
    print(f"brute_force_laps = {method_state['brute_force_laps']}")
    print(f"k_select_laps = {method_state['k_select_laps']}")
    print(f"zoom_radius_laps = {method_state['zoom_radius_laps']}")
    print(f"relax_max_scf_calls = {method_state.get('relax_max_scf_calls')}")
    print(f"concours_enabled = {concours_enabled(method_state)}")

    relax_cache_state = cache_state_for_relax(method_state)

    if method == "wave1":
        raw_extra = {
            "prefix": prefix,
            "n_interpolated": method_state["n_interpolated"],
        }

        raw_is_valid = folder_is_valid_or_trusted(
            raw_folder,
            state=relax_cache_state,
            kind="relax_raw",
            method=method,
            trust_key="trust_existing_raw_npz",
            extra=raw_extra,
        )

        if not raw_is_valid:
            run_interpolated_relaxation_with_optional_rigid_groups(
                state=method_state,
                cfg=cfg,
                raw_folder=raw_folder,
            )

            write_metadata(
                raw_folder,
                state=relax_cache_state,
                kind="relax_raw",
                method=method,
                extra=raw_extra,
            )
        else:
            print(f"{method} raw relaxation already available: {raw_folder}")

    else:
        if source_npz is None:
            raise ValueError("wave2 needs source_npz")

        source_npz = Path(source_npz)
        source = load_chain(source_npz)
        atoms = source["atoms"]

        target_nodes = int(method_state["n_interpolated"]) + 2
        geoms = uniformly_resample_path(source["geoms"], target_nodes=target_nodes)

        raw_extra = {
            "prefix": prefix,
            "source_npz": file_fingerprint(source_npz),
            "n_interpolated": method_state["n_interpolated"],
            "target_nodes": target_nodes,
        }

        raw_is_valid = folder_is_valid_or_trusted(
            raw_folder,
            state=relax_cache_state,
            kind="relax_raw",
            method=method,
            trust_key="trust_existing_raw_npz",
            extra=raw_extra,
        )

        if not raw_is_valid:
            run_given_relaxation_with_optional_rigid_groups(
                atoms=atoms,
                geoms=geoms,
                state=method_state,
                cfg=cfg,
                raw_folder=raw_folder,
            )

            write_metadata(
                raw_folder,
                state=relax_cache_state,
                kind="relax_raw",
                method=method,
                extra=raw_extra,
            )
        else:
            print(f"{method} raw relaxation already available: {raw_folder}")

    clean_extra = {"prefix": prefix}

    clean_is_valid = folder_is_valid_or_trusted(
        clean_folder,
        state=relax_cache_state,
        kind="relax_clean",
        method=method,
        trust_key="trust_existing_clean_npz",
        extra=clean_extra,
    )

    if not clean_is_valid:
        regenerate_raw_npz_to_clean(prefix, cfg)

        write_metadata(
            clean_folder,
            state=relax_cache_state,
            kind="relax_clean",
            method=method,
            extra=clean_extra,
        )
    else:
        print(f"{method} clean relaxation already available: {clean_folder}")

    best = brute_search(method_state, raw_folder, clean_folder)

    npz_data = load_all_npz_dict(clean_folder)

    if not npz_data:
        raise ValueError(f"No npz files found in clean folder: {clean_folder}")

    atoms = next(iter(npz_data.values()))["atoms"]

    if concours_enabled(method_state):
        chosen = choose_best_reparam_candidate(
            archive=best["archive"],
            atoms=atoms,
            cfg=cfg,
            state=method_state,
            method=method,
        )

        chosen_entry = chosen["entry"]

        print_origins(
            f"{method_label.upper()} selected chain origins",
            chosen_entry["pool"],
            chosen_entry["combo"],
        )

        save_chain_summary(
            output_folder=wave_folder,
            stem=stem,
            atoms=atoms,
            geoms=chosen["geoms"],
            energies_abs=chosen["energies_abs"],
        )
    else:
        print("[BASIC] Champion Arena muted. Using best brute-score chain and reparametrizing to target_neb_beads.")

        print_origins(
            f"{method_label.upper()} best brute-score chain origins",
            best["pool"],
            best["combo"],
        )

        geoms, energies_abs = make_basic_guess_from_best(
            best,
            target_nodes=int(method_state["target_neb_beads"]),
        )

        save_chain_summary(
            output_folder=wave_folder,
            stem=stem,
            atoms=atoms,
            geoms=geoms,
            energies_abs=energies_abs,
        )

    if method == "wave1":
        state["wave1_primary_chain_npz"] = target_npz

    state["final_guess_npzs"][method_label] = target_npz

    return target_npz


def run_custom_guess_protocol(state: Dict[str, Any]) -> Dict[str, Any]:
    methods = list(state.get("custom_guess_methods", []))
    state["final_guess_npzs"] = {}

    skipped = [m for m in methods if m not in SUPPORTED_METHODS]

    if skipped:
        print(f"Skipping removed/unknown custom guess methods: {skipped}")

    need_wave1 = any(m in methods for m in ("wave1", "wave2"))

    if need_wave1:
        wave1_npz = run_wave_search(state, "wave1")

        if "wave1" not in methods:
            wave1_label = guess_method_label(make_method_state(state, "wave1"), "wave1")
            state["final_guess_npzs"].pop(wave1_label, None)

    wave1_npz = state.get("wave1_primary_chain_npz")

    if "wave2" in methods:
        if wave1_npz is None:
            raise ValueError("wave2 requested, but wave1_primary_chain_npz is missing")

        run_wave_search(
            state,
            "wave2",
            source_npz=wave1_npz,
        )

    return state


def neb_done(state: Dict[str, Any], run_name: str) -> bool:
    neb_folder = state["base_cfg"].neb_folder
    result_json = neb_folder / f"neb_{run_name}" / f"{run_name}_results.json"
    summary_npz = neb_folder / f"neb_{run_name}" / f"{run_name}_summary.npz"

    return result_json.exists() or summary_npz.exists()


def run_standard_neb(state: Dict[str, Any]):
    if standard_neb_muted(state):
        print("\n>>> Standard NEB muted. Skipping standard NEB.")
        return None

    run_name = standard_run_name(state)
    state["standard_neb_run_name"] = run_name

    if neb_done(state, run_name):
        print(f"Standard NEB already done: {run_name}")
        return run_name

    print("\n>>> Running standard guess protocol...")

    neb_wrapper(state, run_name=run_name)

    return run_name


def run_custom_nebs(state: Dict[str, Any]):
    print("\n>>> Running custom guess protocol...")
    state = run_custom_guess_protocol(state)

    print("\n>>> Final guesses found:")

    for method_name, npz_path in state.get("final_guess_npzs", {}).items():
        print(f"  {method_name} -> {npz_path}")

    if all_neb_muted(state):
        print("\n>>> All NEB runs are muted. Guesses prepared; custom NEB runs skipped.")
        state["custom_neb_run_names"] = []
        return state

    custom_run_names = []

    for method_name, npz_path in state.get("final_guess_npzs", {}).items():
        if npz_path is None:
            print(f"Skipping {method_name}: no guess npz")
            continue

        npz_path = Path(npz_path)

        if not npz_path.exists():
            print(f"Guess file missing for {method_name}: {npz_path}")
            continue

        run_name = custom_run_name(state, method_name)

        if neb_done(state, run_name):
            print(f"Custom NEB already done: {run_name}")
            custom_run_names.append(run_name)
            continue

        chain_data = load_chain(npz_path)
        final_geoms = chain_data["geoms"]
        target_beads = int(state["target_neb_beads"])

        if len(final_geoms) != target_beads:
            print(
                f"   [WARN] Guess has {len(final_geoms)} beads, expected {target_beads}. "
                f"Using keep-original densify fallback."
            )

            final_geoms, _ = densify_to_target_keep_originals_with_mask(
                final_geoms,
                target_beads,
            )
        else:
            print_chain_quality(
                f"{method_name} before NEB",
                final_geoms,
                chain_data.get("energies_abs", None),
            )

        print(f"\n>>> Starting NEB for method: {method_name}")
        print(f">>> Run name: {run_name}")

        neb_wrapper(
            state,
            atoms=chain_data["atoms"],
            myguess=final_geoms,
            run_name=run_name,
        )

        custom_run_names.append(run_name)

    state["custom_neb_run_names"] = custom_run_names

    return state


def classify_standard_if_available(state: Dict[str, Any], standard_name: str | None):
    if standard_name is None:
        state["standard_neb_info"] = {
            "standard_neb_muted": True,
        }
        return state

    try:
        standard_info = classify_standard_neb(state)
    except Exception as exc:
        print(f"[WARN] Could not classify standard NEB: {exc}")
        standard_info = {
            "classification_failed": True,
            "error": str(exc),
        }

    state["standard_neb_info"] = standard_info

    if state.get("GALIMA_NUSIRASYNETI", False):
        state = apply_dynamic_guess_budget(state, standard_info)

    print("\n>>> Standard NEB reconnaissance:")

    for k, v in standard_info.items():
        print(f"  {k}: {v}")

    return state


def generate_final_neb_report(state: Dict[str, Any], standard_name: str | None):
    if all_neb_muted(state):
        print("\n>>> All NEB runs are muted. Final NEB report skipped.")
        return

    custom_runs = list(state.get("custom_neb_run_names", []))

    if standard_name is None:
        print("\n>>> Standard NEB was muted. Combined standard-vs-custom report skipped.")
        return

    if not custom_runs:
        print("\n>>> No custom NEB runs found. Combined report skipped.")
        return

    print("\n>>> Generating final combined NEB comparison report...")

    try:
        compare_all_neb(
            state["base_cfg"].neb_folder,
            standard_name,
            custom_runs,
            state["pair_tag"],
            force=True,
        )
    except TypeError:
        compare_all_neb(state)


def run_pair(state: Dict[str, Any]):
    print(f"\n\n==============================")
    print(f">>> Running pair: {state['pair_tag']}")
    print(f"==============================")

    if not state["xyz_start"].exists() or not state["xyz_end"].exists():
        stage_0_align(state)
    else:
        print("Prepared endpoints already exist. Skipping alignment.")

    standard_name = run_standard_neb(state)

    state = classify_standard_if_available(state, standard_name)
    state = run_custom_nebs(state)

    generate_final_neb_report(state, standard_name)

    return state


def make_pairs(points):
    return list(zip(points[:-1], points[1:]))


def run_multisegment_job(
    *,
    job,
    base_run_name: str,
    points,
    interpolation: str = "linear",
    overrides: dict | None = None,
):
    final_states = {}

    for (start_label, start_raw), (end_label, end_raw) in make_pairs(points):
        pair_tag = f"{start_label}-{end_label}"

        state = make_pipeline_config(
            job=job,
            base_run_name=base_run_name,
            pair_tag=pair_tag,
            start_raw=start_raw,
            end_raw=end_raw,
            interpolation=interpolation,
            overrides=overrides,
        )

        final_state = run_pair(state)
        final_states[pair_tag] = final_state

    return final_states