import numpy as np
from pathlib import Path

from utils.io.npz_io import load_all_npz_dict


DEFAULT_UPHILL_SCHEDULE = (
    (1e-6, 90),
    (1e-5, 75),
    (1e-4, 40),
    (1e-3, 15),
)


def select_non_uphill_points(energies, allow_uphill):
    energies = np.asarray(energies, dtype=float)
    mask = np.zeros(len(energies), dtype=bool)

    if len(energies) == 0:
        return mask

    mask[0] = True
    last_e = energies[0]

    for i in range(1, len(energies)):
        if energies[i] <= last_e + allow_uphill:
            mask[i] = True
            last_e = energies[i]

    mask[int(np.argmin(energies))] = True

    return mask


def physical_mask_by_gradient_norm(grads, max_force_threshold=50.0):
    grads = np.asarray(grads, dtype=float)
    force_norms = np.linalg.norm(grads, axis=(1, 2))

    return np.isfinite(force_norms) & (force_norms < max_force_threshold)


def rescue_points_by_small_jumps(energies, mask, min_keep=10, max_jump=0.01, verbose=False):
    n_kept = int(np.sum(mask))

    if n_kept >= min_keep:
        return mask

    rejected_idx = np.where(~mask)[0]

    if len(rejected_idx) == 0:
        return mask

    jumps = []

    for idx in rejected_idx:
        min_before = np.min(energies[:idx + 1])
        jumps.append(energies[idx] - min_before)

    jumps = np.asarray(jumps)

    valid = (jumps > 0.0) & (jumps < max_jump)
    valid_idx = rejected_idx[valid]
    valid_jumps = jumps[valid]

    need = min_keep - n_kept

    if len(valid_idx) > 0 and need > 0:
        rescued = valid_idx[np.argsort(valid_jumps)[:need]]
        mask[rescued] = True

        if verbose:
            print(f"  [RESCUE] added {len(rescued)} frames; kept {int(np.sum(mask))}")
    elif verbose:
        print(f"  [RESCUE FAIL] kept {n_kept}, needed {min_keep}")

    return mask


def clean_trajectory(
    data,
    max_force=50.0,
    uphill_schedule=DEFAULT_UPHILL_SCHEDULE,
    rescue_min_keep=10,
    rescue_max_jump=0.01,
    verbose=False,
):
    geoms = np.asarray(data["geoms"])
    energies = np.asarray(data["energies"], dtype=float)
    grads = np.asarray(data["grads"])
    atoms = data["atoms"]

    original_len = len(energies)
    original_ids = np.arange(original_len, dtype=int)

    if original_len == 0:
        return {
            "atoms": atoms,
            "geoms": geoms,
            "energies": energies,
            "grads": grads,
            "ids": original_ids,
        }

    phys_mask = physical_mask_by_gradient_norm(
        grads,
        max_force_threshold=max_force,
    )

    geoms = geoms[phys_mask]
    energies = energies[phys_mask]
    grads = grads[phys_mask]
    ids = original_ids[phys_mask]

    if len(energies) == 0:
        if verbose:
            print("  [WARN] all frames removed by force filter; keeping first raw frame")

        return {
            "atoms": atoms,
            "geoms": np.asarray(data["geoms"])[:1],
            "energies": np.asarray(data["energies"], dtype=float)[:1],
            "grads": np.asarray(data["grads"])[:1],
            "ids": original_ids[:1],
        }

    mono_mask = None

    for allow_uphill, target_keep in uphill_schedule:
        mono_mask = select_non_uphill_points(
            energies,
            allow_uphill=allow_uphill,
        )

        n_kept = int(np.sum(mono_mask))

        if verbose:
            print(
                f"  allow_uphill={allow_uphill:.0e}: "
                f"kept {n_kept}/{len(energies)} "
                f"(target {target_keep})"
            )

        if n_kept >= target_keep:
            break

    mono_mask = rescue_points_by_small_jumps(
        energies=energies,
        mask=mono_mask,
        min_keep=rescue_min_keep,
        max_jump=rescue_max_jump,
        verbose=verbose,
    )

    cleaned = {
        "atoms": atoms,
        "geoms": geoms[mono_mask],
        "energies": energies[mono_mask],
        "grads": grads[mono_mask],
        "ids": ids[mono_mask],
    }

    if verbose:
        final_len = len(cleaned["energies"])
        reduction = 100.0 * (original_len - final_len) / original_len
        print(f"  Steps: {original_len} -> {final_len} (-{reduction:.1f}%)")

    return cleaned


def regenerate_npz_without_outliers(
    src_folder,
    dst_folder,
    verbose=True,
    max_force=50.0,
    rescue_min_keep=10,
    rescue_max_jump=0.01,
):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    raw_npz = load_all_npz_dict(folder=src_folder)

    for filename, data in raw_npz.items():
        if verbose:
            print(f"\nCleaning: {filename}")

        clean_data = clean_trajectory(
            data=data,
            max_force=max_force,
            rescue_min_keep=rescue_min_keep,
            rescue_max_jump=rescue_max_jump,
            verbose=verbose,
        )

        np.savez_compressed(
            dst_folder / filename,
            atoms=np.asarray(clean_data["atoms"]),
            geoms=np.asarray(clean_data["geoms"]),
            energies=np.asarray(clean_data["energies"]),
            grads=np.asarray(clean_data["grads"]),
            ids=np.asarray(clean_data["ids"]),
        )

    return dst_folder


def regenerate_raw_npz_to_clean(name, cfg, verbose=True):
    src_folder = Path(cfg.opt_folder) / f"{name}_raw"
    dst_folder = Path(cfg.opt_folder) / f"{name}_clean"

    return regenerate_npz_without_outliers(
        src_folder=src_folder,
        dst_folder=dst_folder,
        verbose=verbose,
        max_force=getattr(cfg, "max_force_threshold", 50.0),
        rescue_min_keep=getattr(cfg, "clean_rescue_min_keep", 10),
        rescue_max_jump=getattr(cfg, "clean_rescue_max_jump", 0.01),
    )